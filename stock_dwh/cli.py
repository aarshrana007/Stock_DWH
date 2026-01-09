from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

from stock_dwh.config import get_paths, get_sources, get_run_config
from stock_dwh.meta import Watermarks
from stock_dwh.utils.logging import get_logger
from stock_dwh.utils.io import write_parquet, read_parquet, write_json

from stock_dwh.ingest.news import (
    load_news_csv,
    load_rss,
    normalize_news,
    filter_incremental as news_filter,
    update_last_seen as news_last,
)

from stock_dwh.ingest.market import (
    load_market_csv,
    filter_incremental as mkt_filter,
    update_last_seen as mkt_last,
)

from stock_dwh.transform.silver import (
    silver_fact_news,
    silver_fact_prices,
)

from stock_dwh.features.sentiment import finbert_placeholder
from stock_dwh.features.build import build_features
from stock_dwh.model.train import train_placeholder
from stock_dwh.model.infer import load_model, predict


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def wh_join(base, *parts):
    """Join paths for local FS or S3."""
    if isinstance(base, str):
        return "/".join([base, *parts])
    return base.joinpath(*parts)


def write_partitioned(df, root, partition_col="dt", filename="part.parquet"):
    if df.empty:
        return

    if partition_col not in df.columns:
        write_parquet(df, wh_join(root, filename))
        return

    for val, chunk in df.groupby(partition_col):
        out = wh_join(root, f"{partition_col}={val}", filename)
        write_parquet(chunk.reset_index(drop=True), out)


# ---------------------------------------------------------------------
# INGEST
# ---------------------------------------------------------------------
def ingest():
    paths = get_paths()
    src = get_sources()
    rc = get_run_config()
    log = get_logger("stock_dwh.ingest", wh_join(paths.logs, "ingest.log"))

    wm_path = wh_join(paths.repo_root, rc.watermark_path)
    wm = Watermarks.load(wm_path)

    # ---------------- NEWS ----------------
    news_frames = []

    csv_path = Path(src.news_github_csv_path)
    if csv_path.exists():
        log.info(f"Loading news CSV: {csv_path}")
        news_frames.append(load_news_csv(csv_path))

    if src.news_rss_urls:
        log.info(f"Loading RSS: {len(src.news_rss_urls)} urls")
        news_frames.append(load_rss(src.news_rss_urls))

    raw_news = (
        pd.concat(news_frames, ignore_index=True)
        if news_frames
        else pd.DataFrame(columns=["title", "link", "published", "summary", "source"])
    )

    raw_news = normalize_news(raw_news)
    raw_news = news_filter(raw_news, wm.news_last_seen_ts)

    write_partitioned(raw_news, wh_join(paths.bronze, "raw_news"))
    wm.news_last_seen_ts = news_last(raw_news, wm.news_last_seen_ts)

    # ---------------- MARKET ----------------
    raw_px = pd.DataFrame()
    mkt_path = Path(src.market_csv_path)

    if mkt_path.exists():
        log.info(f"Loading market CSV: {mkt_path}")
        raw_px = load_market_csv(mkt_path)

        # REQUIRED NORMALIZATION (your CSV has `datetime`)
        raw_px["ts_utc"] = pd.to_datetime(raw_px["datetime"], utc=True)
        raw_px["dt"] = raw_px["ts_utc"].dt.date.astype(str)

        raw_px = mkt_filter(raw_px, wm.market_last_seen_ts)

        write_partitioned(raw_px, wh_join(paths.bronze, "raw_prices"))
        wm.market_last_seen_ts = mkt_last(raw_px, wm.market_last_seen_ts)
    else:
        log.warning(f"Market CSV not found at {mkt_path}")

    wm.save(wm_path)
    log.info("Ingest done.")


# ---------------------------------------------------------------------
# SILVER
# ---------------------------------------------------------------------
def silver():
    paths = get_paths()
    log = get_logger("stock_dwh.silver", wh_join(paths.logs, "silver.log"))

    raw_news = read_parquet(wh_join(paths.bronze, "raw_news"))
    raw_px = read_parquet(wh_join(paths.bronze, "raw_prices"))

    fact_news = silver_fact_news(raw_news) if not raw_news.empty else pd.DataFrame()
    fact_px = silver_fact_prices(raw_px) if not raw_px.empty else pd.DataFrame()

    write_partitioned(fact_news, wh_join(paths.silver, "fact_news"))
    write_partitioned(fact_px, wh_join(paths.silver, "fact_prices"))

    log.info("Silver build done.")


# ---------------------------------------------------------------------
# INFER (RESTORES ALL 50 STOCKS)
# ---------------------------------------------------------------------
def infer():
    paths = get_paths()
    log = get_logger("stock_dwh.infer", wh_join(paths.logs, "infer.log"))

    fact_news = read_parquet(wh_join(paths.silver, "fact_news"))
    fact_px = read_parquet(wh_join(paths.silver, "fact_prices"))

    universe = pd.read_csv("Data/market/nifty50.csv")
    universe["ticker"] = universe["ticker"].str.upper()

    if fact_px.empty:
        log.warning("No prices found in silver.")
        preds = universe.copy()
        preds["pred"] = None
        preds["signal_status"] = "NO_PRICE"
    else:
        fact_px["ticker"] = fact_px["ticker"].str.upper()
        fact_px = universe.merge(fact_px, on="ticker", how="left")
        fact_px["has_price"] = fact_px["close"].notna()

        scored = finbert_placeholder(fact_news) if not fact_news.empty else pd.DataFrame()

        valid_px = fact_px[fact_px["has_price"]].copy()

        if valid_px.empty:
            preds = universe.copy()
            preds["pred"] = None
            preds["signal_status"] = "NO_PRICE"
        else:
            asof = valid_px["ts_utc"].max()
            feats = build_features(valid_px, scored, asof)

            model_path = wh_join(paths.artifacts, "models/champion/model.pkl")
            model = load_model(model_path)
            preds_valid = predict(model, feats)

            preds_valid["signal_status"] = "MODEL"
            preds = universe.merge(preds_valid, on="ticker", how="left")
            preds.loc[preds["pred"].isna(), "signal_status"] = "NO_PRICE"

    preds["dt"] = pd.Timestamp.utcnow().date().astype(str)

    write_partitioned(preds, wh_join(paths.gold, "fact_predictions"))

    ranked = preds[preds["pred"].notna()].sort_values("pred", ascending=False)
    top = ranked.head(10).assign(bucket="TOP")
    bot = ranked.tail(10).assign(bucket="BOTTOM")
    mart = pd.concat([top, bot], ignore_index=True)

    write_partitioned(
        mart,
        wh_join(paths.gold, "mart_market_snapshot_topbottom"),
        filename="snapshot.parquet",
    )

    log.info("Inference done.")


# ---------------------------------------------------------------------
# TRAIN (OPTIONAL)
# ---------------------------------------------------------------------
def train():
    paths = get_paths()
    log = get_logger("stock_dwh.train", wh_join(paths.logs, "train.log"))

    fact_px = read_parquet(wh_join(paths.silver, "fact_prices"))
    if fact_px.empty:
        log.warning("No prices found. Cannot train.")
        return

    fact_px["next_close"] = fact_px.groupby("ticker")["close"].shift(-1)
    fact_px["target"] = (
        fact_px["next_close"] / fact_px["close"] - 1.0
    ).fillna(0.0)

    training = fact_px[["ticker", "target"]].dropna()

    model_path = wh_join(paths.artifacts, "models/champion/model.pkl")
    meta = train_placeholder(training, "target", model_path)
    write_json(meta, model_path.replace(".pkl", ".json"))

    log.info("Training done.")


# ---------------------------------------------------------------------
# CLI ENTRYPOINT (THIS WAS MISSING BEFORE)
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(prog="stock_dwh")
    parser.add_argument("cmd", choices=["ingest", "silver", "infer", "train"])
    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest()
    elif args.cmd == "silver":
        silver()
    elif args.cmd == "infer":
        infer()
    elif args.cmd == "train":
        train()


if __name__ == "__main__":
    main()
