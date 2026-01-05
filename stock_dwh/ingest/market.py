from __future__ import annotations

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging

log = logging.getLogger(__name__)

BRONZE_DIR = Path("warehouse/bronze/raw_prices")
STATE_FILE = Path("warehouse/bronze/_market_last_seen.txt")

# ---------------------------------------------------------
# NIFTY 50 Yahoo Finance tickers
# ---------------------------------------------------------
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

# =========================================================
# CLI EXPECTED FUNCTIONS (SIGNATURES MUST MATCH)
# =========================================================

def load_market_csv(mkt_path: str | Path) -> pd.DataFrame:
    """
    CLI entrypoint.
    CSV path is ignored; Yahoo Finance is used instead.
    """
    log.info("Loading market CSV: %s", mkt_path)

    end = datetime.now(timezone.utc).date()
    start = get_last_seen() or (end - timedelta(days=730))

    records = []

    for ticker in NIFTY50:
        try:
            df = yf.download(
                ticker,
                start=str(start),
                end=str(end + timedelta(days=1)),
                interval="1d",
                auto_adjust=False,
                progress=False
            )

            if df.empty:
                log.warning("No data for %s", ticker)
                continue

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df["ticker"] = ticker.replace(".NS", "")

            records.append(df)

        except Exception as e:
            log.warning("Skipping %s due to error: %s", ticker, e)

    if not records:
        return pd.DataFrame()

    out = pd.concat(records, ignore_index=True)

    out = out.rename(columns={"date": "ts"})
    out["ts_utc"] = pd.to_datetime(out["ts"], utc=True)
    out["dt"] = out["ts_utc"].dt.date

    return out[[
        "ticker",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ts_utc",
        "dt"
    ]]


def filter_incremental(
    df: pd.DataFrame,
    last_seen_ts: pd.Timestamp | None
) -> pd.DataFrame:
    """
    CLI passes last_seen_ts.
    Filter rows strictly newer than last ingest.
    """
    if df.empty or last_seen_ts is None:
        return df

    return df[df["ts_utc"] > last_seen_ts]


def update_last_seen(df: pd.DataFrame) -> None:
    """
    Persist max ts_utc for next incremental run.
    """
    if df.empty:
        return

    last_ts = df["ts_utc"].max()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(last_ts.isoformat())


# =========================================================
# HELPERS
# =========================================================

def get_last_seen():
    if not STATE_FILE.exists():
        return None
    return pd.to_datetime(STATE_FILE.read_text(), utc=True)
