from __future__ import annotations

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging

log = logging.getLogger(__name__)

# --------------------------------------------------
# NIFTY 50 Yahoo Finance tickers
# --------------------------------------------------
NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "LT.NS", "SBIN.NS", "AXISBANK.NS", "HINDUNILVR.NS", "ITC.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "KOTAKBANK.NS", "HCLTECH.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "NTPC.NS", "POWERGRID.NS",
    "ONGC.NS", "TITAN.NS", "ULTRACEMCO.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "COALINDIA.NS", "WIPRO.NS", "ASIANPAINT.NS",
    "JSWSTEEL.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "NESTLEIND.NS",
    "BPCL.NS", "GRASIM.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "DIVISLAB.NS", "BRITANNIA.NS", "HINDALCO.NS", "CIPLA.NS",
    "DRREDDY.NS", "TECHM.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
    "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "INDUSINDBK.NS",
    "UPL.NS", "LTIM.NS", "SHRIRAMFIN.NS", "M&M.NS"
]

# ==================================================
# CLI EXPECTED FUNCTIONS (DO NOT CHANGE SIGNATURES)
# ==================================================

def load_market_csv(mkt_path: str | Path) -> pd.DataFrame:
    """
    CLI entry point.
    CSV path is ignored; Yahoo Finance is used instead.
    """
    log.info("Loading market CSV: %s", mkt_path)

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=730)

    frames = []

    for ticker in NIFTY50:
        try:
            df = yf.download(
                ticker,
                start=str(start),
                end=str(end + timedelta(days=1)),
                interval="1d",
                auto_adjust=False,
                progress=False,
            )

            if df.empty:
                log.warning("No data for %s", ticker)
                continue

            df = df.reset_index()

            # 🔧 FIX: flatten tuple columns safely
            df.columns = [
                c[0].lower() if isinstance(c, tuple) else c.lower()
                for c in df.columns
            ]

            df["ticker"] = ticker.replace(".NS", "")
            df.rename(columns={"date": "ts"}, inplace=True)

            df["ts_utc"] = pd.to_datetime(df["ts"], utc=True)
            df["dt"] = df["ts_utc"].dt.date

            frames.append(df)

        except Exception as e:
            log.warning("Skipping %s due to error: %s", ticker, e)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    return out[[
        "ticker",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ts_utc",
        "dt",
    ]]


def filter_incremental(
    df: pd.DataFrame,
    last_seen_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    """
    Keep only rows newer than last ingested timestamp.
    """
    if df.empty or last_seen_ts is None:
        return df

    return df[df["ts_utc"] > last_seen_ts]


def update_last_seen(
    df: pd.DataFrame,
    prev_last_seen_ts: pd.Timestamp | None,
) -> pd.Timestamp | None:
    """
    CLI expects a RETURN value.
    """
    if df.empty:
        return prev_last_seen_ts

    return df["ts_utc"].max()
