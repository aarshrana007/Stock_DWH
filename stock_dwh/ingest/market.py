from __future__ import annotations

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta, timezone

BRONZE_DIR = Path("warehouse/bronze/raw_prices")

# NIFTY 50 Yahoo Finance tickers
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

def fetch_ohlcv(
    tickers: list[str],
    start: str,
    end: str
) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance"""
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True
    )

    records = []

    for ticker in tickers:
        if ticker not in df.columns.levels[0]:
            continue

        tdf = df[ticker].reset_index()
        tdf.columns = [c.lower() for c in tdf.columns]
        tdf["ticker"] = ticker.replace(".NS", "")
        records.append(tdf)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def run():
    print("📈 Fetching NIFTY 50 daily OHLCV")

    # Fetch last 2 years (safe demo default)
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=730)

    df = fetch_ohlcv(
        tickers=NIFTY50,
        start=str(start),
        end=str(end)
    )

    if df.empty:
        print("⚠️ No market data fetched")
        return

    # Normalize schema
    df = df.rename(columns={
        "date": "ts",
        "adj close": "adj_close"
    })

    df["ts_utc"] = pd.to_datetime(df["ts"], utc=True)
    df["dt"] = df["ts_utc"].dt.date

    df = df[[
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

    # Write partitioned parquet
    for dt, part in df.groupby("dt"):
        out = BRONZE_DIR / f"dt={dt}"
        out.mkdir(parents=True, exist_ok=True)
        part.to_parquet(out / "part.parquet", index=False)

    print(f"✅ Ingested {df['ticker'].nunique()} stocks for {df['dt'].nunique()} days")
