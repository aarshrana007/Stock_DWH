from __future__ import annotations

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta, timezone

BRONZE_DIR = Path("warehouse/bronze/raw_prices")
STATE_FILE = Path("warehouse/bronze/_market_last_seen.txt")

# NIFTY 50 Yahoo tickers
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

# -------------------------------------------------------------------
# CLI EXPECTED FUNCTIONS (DO NOT REMOVE)
# -------------------------------------------------------------------

def load_market_csv() -> pd.DataFrame:
    """
    CLI entrypoint.
    Replaced CSV logic with Yahoo Finance fetch.
    """
    end = datetime.now(timezone.utc).date()
    start = get_last_seen() or (end - timedelta(days=730))

    df = yf.download(
        tickers=NIFTY50,
        start=str(start),
        end=str(end + timedelta(days=1)),
        interval="1d",
        group_by="ticker",
        threads=True,
        auto_adjust=False
    )

    records = []
    for ticker in NIFTY50:
        if ticker not in df.columns.levels[0]:
            continue

        tdf = df[ticker].reset_index()
        tdf.columns = [c.lower() for c in tdf.columns]
        tdf["ticker"] = ticker.replace(".NS", "")
        records.append(tdf)

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


def filter_incremental(df: pd.DataFrame) -> pd.DataFrame:
    """No-op incremental filter (Yahoo already date bounded)"""
    return df


def update_last_seen(df: pd.DataFrame) -> None:
    """Persist max ingested date"""
    if df.empty:
        return
    last_dt = df["dt"].max()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(str(last_dt))


# -------------------------------------------------------------------
# MAIN INGEST LOGIC (called by CLI)
# -------------------------------------------------------------------

def run():
    df = load_market_csv()
    df = filter_incremental(df)

    if df.empty:
        print("⚠️ No new market data")
        return

    for dt, part in df.groupby("dt"):
        out = BRONZE_DIR / f"dt={dt}"
        out.mkdir(parents=True, exist_ok=True)
        part.to_parquet(out / "part.parquet", index=False)

    update_last_seen(df)

    print(
        f"✅ Ingested {df['ticker'].nunique()} stocks "
        f"for {df['dt'].nunique()} days"
    )


# -------------------------------------------------------------------

def get_last_seen():
    if not STATE_FILE.exists():
        return None
    return pd.to_datetime(STATE_FILE.read_text()).date()
