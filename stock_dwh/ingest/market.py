from __future__ import annotations

import pandas as pd
import requests
import zipfile
import io
from datetime import datetime, timedelta
from pathlib import Path
import logging

log = logging.getLogger(__name__)

# --------------------------------------------------
# NIFTY 50 symbols
# --------------------------------------------------
NIFTY50 = {
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","LT","SBIN","AXISBANK",
    "HINDUNILVR","ITC","BAJFINANCE","BAJAJFINSV","KOTAKBANK","HCLTECH",
    "MARUTI","SUNPHARMA","NTPC","POWERGRID","ONGC","TITAN",
    "ULTRACEMCO","ADANIENT","ADANIPORTS","COALINDIA","WIPRO",
    "ASIANPAINT","JSWSTEEL","TATAMOTORS","TATASTEEL","NESTLEIND",
    "BPCL","GRASIM","HDFCLIFE","SBILIFE","DIVISLAB","BRITANNIA",
    "HINDALCO","CIPLA","DRREDDY","TECHM","HEROMOTOCO","EICHERMOT",
    "APOLLOHOSP","BAJAJ-AUTO","INDUSINDBK","UPL","LTIM",
    "SHRIRAMFIN","M&M"
}

# ✅ CORRECT NSE Equity Bhavcopy URL
NSE_URL = "https://archives.nseindia.com/content/historical/EQUITIES/{year}/{mon}/cm{dd}{mon}{year}bhav.csv.zip"

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ==================================================
# CLI EXPECTED FUNCTIONS
# ==================================================

def load_market_csv(mkt_path: str | Path) -> pd.DataFrame:
    """Load OHLCV market data from a local CSV.

    Expected input columns (case-insensitive):
      - ticker (or symbol)
      - datetime / ts_utc / timestamp / date
      - open, high, low, close, volume

    Produces:
      ticker, ts_utc (UTC), open, high, low, close, volume, dt (YYYY-MM-DD)
    """
    p = Path(mkt_path)
    df = pd.read_csv(p)

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # ticker
    if "ticker" not in df.columns:
        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "ticker"})
        else:
            raise ValueError(f"Market CSV missing ticker/symbol column: {p}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # find time column
    time_col = None
    for cand in ("ts_utc", "datetime", "timestamp", "date", "time"):
        if cand in df.columns:
            time_col = cand
            break

    if time_col is None:
        # Try common variants
        for cand in ("date_time", "datetimestamp", "trade_date"):
            if cand in df.columns:
                time_col = cand
                break

    if time_col is None:
        # Don't crash the pipeline; return empty and let caller log a warning
        log.warning("Market CSV has no datetime column. Columns=%s", df.columns.tolist())
        return pd.DataFrame(columns=["ticker","ts_utc","open","high","low","close","volume","dt"])

    # parse to UTC
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.assign(ts_utc=ts)
    df = df.dropna(subset=["ts_utc"])

    # numeric columns
    for c in ("open","high","low","close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.NA

    df["dt"] = df["ts_utc"].dt.date.astype(str)

    out = df[["ticker","ts_utc","open","high","low","close","volume","dt"]].copy()
    return out


def filter_incremental(df, last_seen_ts):
    if df.empty or last_seen_ts is None:
        return df
    return df[df["ts_utc"] > last_seen_ts]


def update_last_seen(df, prev_last_seen_ts):
    if df.empty:
        return prev_last_seen_ts
    return df["ts_utc"].max()

# ==================================================
# HELPERS
# ==================================================

def daterange(start, end):
    for n in range((end - start).days + 1):
        yield start + timedelta(days=n)
