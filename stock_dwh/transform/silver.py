from __future__ import annotations
import pandas as pd
from ..utils.dedup import canonicalize_text, sha1_text

def silver_fact_news(raw_news: pd.DataFrame) -> pd.DataFrame:
    """Create a clean fact_news table (Silver).
    Assumes raw_news already has: canonical_id, published_ts, clean_text, title, summary, link, source, dt
    """
    if raw_news.empty:
        return raw_news
    df = raw_news.copy()
    # enforce types
    df["canonical_id"] = df["canonical_id"].astype(str)
    df = df.drop_duplicates(subset=["canonical_id"], keep="last").reset_index(drop=True)
    df["entities"] = None  # placeholder (NER later)
    return df[["canonical_id","published_ts","dt","source","link","title","summary","clean_text","entities"]]

def silver_fact_prices(raw_prices: pd.DataFrame) -> pd.DataFrame:
    if raw_prices.empty:
        return raw_prices
    df = raw_prices.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.sort_values(["ticker","ts_utc"])
    df = df.drop_duplicates(subset=["ticker","ts_utc"], keep="last").reset_index(drop=True)
    return df[["ticker","ts_utc","dt","open","high","low","close","volume"]]
