from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from .config import get_paths, get_sources, get_run_config
from .meta import Watermarks
from .utils.logging import get_logger
from .utils.io import write_parquet, read_parquet, write_json
from .ingest.news import load_news_csv, load_rss, normalize_news, filter_incremental as news_filter, update_last_seen as news_last
from .ingest.market import load_market_csv, filter_incremental as mkt_filter, update_last_seen as mkt_last
from .transform.silver import silver_fact_news, silver_fact_prices
from .features.sentiment import finbert_placeholder
from .features.build import build_features
from .model.train import train_placeholder
from .model.infer import load_model, predict

def _write_partitioned(df: pd.DataFrame, root: Path, partition_col: str = "dt", filename: str = "part.parquet") -> None:
    if df.empty:
        return
    if partition_col not in df.columns:
        p = root / filename
        write_parquet(df, p)
        return
    for dt, chunk in df.groupby(partition_col):
        p = root / f"{partition_col}={dt}" / filename
        write_parquet(chunk.reset_index(drop=True), p)

def ingest() -> None:
    paths = get_paths()
    src = get_sources()
    rc = get_run_config()
    log = get_logger("stock_dwh.ingest", paths.logs / "ingest.log")

    wm_path = paths.repo_root / rc.watermark_path
    wm_path.parent.mkdir(parents=True, exist_ok=True)
    wm = Watermarks.load(wm_path)

    # NEWS
    news_frames = []
    csv_path = (paths.repo_root / src.news_github_csv_path)
    if csv_path.exists():
        log.info(f"Loading news CSV: {csv_path}")
        news_frames.append(load_news_csv(csv_path))
    if src.news_rss_urls:
        log.info(f"Loading RSS: {len(src.news_rss_urls)} urls")
        news_frames.append(load_rss(src.news_rss_urls))
    raw_news = pd.concat(news_frames, ignore_index=True) if news_frames else pd.DataFrame(columns=["title","link","published","summary","source"])
    raw_news = normalize_news(raw_news)
    raw_news = news_filter(raw_news, wm.news_last_seen_ts)

    _write_partitioned(raw_news, paths.bronze / "raw_news")
    wm.news_last_seen_ts = news_last(raw_news, wm.news_last_seen_ts)

    # MARKET
    mkt_path = (paths.repo_root / src.market_csv_path)
    raw_px = pd.DataFrame()
    if mkt_path.exists():
        log.info(f"Loading market CSV: {mkt_path}")
        raw_px = load_market_csv(mkt_path)
        raw_px = mkt_filter(raw_px, wm.market_last_seen_ts)
        _write_partitioned(raw_px, paths.bronze / "raw_prices")
        wm.market_last_seen_ts = mkt_last(raw_px, wm.market_last_seen_ts)
    else:
        log.warning(f"Market CSV not found at {mkt_path}. Set MARKET_CSV_PATH to your data file.")

    wm.save(wm_path)
    log.info("Ingest done.")

def build_silver() -> None:
    paths = get_paths()
    log = get_logger("stock_dwh.silver", paths.logs / "silver.log")

    # Load all bronze partitions (simple)
    news_files = list((paths.bronze / "raw_news").rglob("*.parquet"))
    px_files = list((paths.bronze / "raw_prices").rglob("*.parquet"))

    raw_news = pd.concat([read_parquet(p) for p in news_files], ignore_index=True) if news_files else pd.DataFrame()
    raw_px = pd.concat([read_parquet(p) for p in px_files], ignore_index=True) if px_files else pd.DataFrame()

    fact_news = silver_fact_news(raw_news) if not raw_news.empty else pd.DataFrame()
    fact_px = silver_fact_prices(raw_px) if not raw_px.empty else pd.DataFrame()

    _write_partitioned(fact_news, paths.silver / "fact_news")
    _write_partitioned(fact_px, paths.silver / "fact_prices")
    log.info("Silver build done.")

def infer_run() -> None:
    paths = get_paths()
    log = get_logger("stock_dwh.infer", paths.logs / "infer.log")

    # load latest silver data
    news_files = list((paths.silver / "fact_news").rglob("*.parquet"))
    px_files = list((paths.silver / "fact_prices").rglob("*.parquet"))
    fact_news = pd.concat([read_parquet(p) for p in news_files], ignore_index=True) if news_files else pd.DataFrame()
    fact_px = pd.concat([read_parquet(p) for p in px_files], ignore_index=True) if px_files else pd.DataFrame()

    if fact_px.empty:
        log.warning("No prices found in silver. Cannot infer.")
        return

    # score sentiment
    scored = finbert_placeholder(fact_news) if not fact_news.empty else pd.DataFrame()

    # pick asof = latest price ts (minus lag)
    fact_px["ts_utc"] = pd.to_datetime(fact_px["ts_utc"], utc=True)
    asof = fact_px["ts_utc"].max()
    feats = build_features(fact_px, scored, asof)

    model_path = paths.artifacts / "models" / "champion" / "model.pkl"
    if not model_path.exists():
        # if no model, create baseline from current returns
        tmp = feats.copy()
        tmp["target"] = tmp["ret_1d"]
        meta = train_placeholder(tmp[["ticker","target"]], "target", model_path)
        write_json(meta, model_path.with_suffix(".json"))
        log.info("Trained baseline model as champion (placeholder).")

    model = load_model(model_path)
    preds = predict(model, feats)
    preds["dt"] = preds["asof_ts"].dt.date.astype(str)

    # gold
    _write_partitioned(preds, paths.gold / "fact_predictions")
    # mart: top/bottom 10
    m = preds.sort_values("pred", ascending=False)
    top = m.head(10).assign(bucket="TOP")
    bot = m.tail(10).assign(bucket="BOTTOM")
    mart = pd.concat([top, bot], ignore_index=True)
    _write_partitioned(mart, paths.gold / "mart_market_snapshot_topbottom", filename="snapshot.parquet")
    log.info("Inference done.")

def train_run() -> None:
    paths = get_paths()
    log = get_logger("stock_dwh.train", paths.logs / "train.log")

    # use gold predictions + prices to create a tiny training set placeholder
    px_files = list((paths.silver / "fact_prices").rglob("*.parquet"))
    fact_px = pd.concat([read_parquet(p) for p in px_files], ignore_index=True) if px_files else pd.DataFrame()
    if fact_px.empty:
        log.warning("No prices found. Cannot train.")
        return
    fact_px["ts_utc"] = pd.to_datetime(fact_px["ts_utc"], utc=True)
    fact_px = fact_px.sort_values(["ticker","ts_utc"])

    # target = next close return (1-step)
    fact_px["next_close"] = fact_px.groupby("ticker")["close"].shift(-1)
    fact_px["target"] = (fact_px["next_close"] / fact_px["close"] - 1.0).fillna(0.0)
    training = fact_px[["ticker","target"]].dropna()

    model_path = paths.artifacts / "models" / "champion" / "model.pkl"
    meta = train_placeholder(training, "target", model_path)
    write_json(meta, model_path.with_suffix(".json"))
    log.info("Training done (placeholder).")

def main():
    ap = argparse.ArgumentParser(prog="stock_dwh")
    ap.add_argument("cmd", choices=["ingest","silver","infer","train"])
    args = ap.parse_args()

    if args.cmd == "ingest":
        ingest()
    elif args.cmd == "silver":
        build_silver()
    elif args.cmd == "infer":
        infer_run()
    elif args.cmd == "train":
        train_run()

if __name__ == "__main__":
    main()
