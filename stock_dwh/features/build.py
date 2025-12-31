from __future__ import annotations
import pandas as pd

def build_features(fact_prices: pd.DataFrame, fact_news_scored: pd.DataFrame, asof_ts: pd.Timestamp) -> pd.DataFrame:
    """Build a simple, PIT-ish feature table for inference.

    For each ticker at asof_ts:
      - last_close
      - return_1d (based on last 2 rows)
      - news_sent_mean_24h
      - news_sent_count_24h
    """
    if fact_prices.empty:
        return pd.DataFrame()

    px = fact_prices.copy()
    px["ts_utc"] = pd.to_datetime(px["ts_utc"], utc=True)
    # px = px[px["ts_utc"] <= asof_ts].sort_values(["ticker","ts_utc"])
    px = (
    px[px["ts_utc"] <= asof_ts]
    .sort_values(["ticker", "ts_utc"])
    .groupby("ticker", as_index=False)
    .tail(2)
    )

    # last 2 per ticker
    last = px.groupby("ticker").tail(2)
    last_close = last.groupby("ticker").tail(1).set_index("ticker")["close"]
    ret_1d = last.groupby("ticker")["close"].apply(lambda s: (s.iloc[-1] / s.iloc[0] - 1.0) if len(s)>=2 else 0.0)

    feats = pd.DataFrame({
        "ticker": last_close.index,
        "asof_ts": asof_ts,
        "last_close": last_close.values,
        "ret_1d": ret_1d.reindex(last_close.index).values,
    })

    if not fact_news_scored.empty:
        nw = fact_news_scored.copy()
        nw["published_ts"] = pd.to_datetime(nw["published_ts"], utc=True)
        w_start = asof_ts - pd.Timedelta(hours=24)
        nw = nw[(nw["published_ts"] > w_start) & (nw["published_ts"] <= asof_ts)]

        # If you already map news->ticker, this will work. If not, it stays empty.
        if "ticker" in nw.columns:
            agg = nw.groupby("ticker").agg(
                news_sent_mean_24h=("sent_score","mean"),
                news_sent_count_24h=("sent_score","count"),
                news_conf_mean_24h=("sent_conf","mean"),
            ).reset_index()
            feats = feats.merge(agg, on="ticker", how="left")

    for c in ["news_sent_mean_24h","news_sent_count_24h","news_conf_mean_24h"]:
        if c in feats.columns:
            feats[c] = feats[c].fillna(0.0)
    return feats
