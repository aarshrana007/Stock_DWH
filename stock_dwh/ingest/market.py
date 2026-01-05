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
    log.info("Loading market CSV: %s", mkt_path)

    end = datetime.today().date()
    start = end - timedelta(days=7)  # incremental window

    session = requests.Session()
    session.headers.update(HEADERS)

    frames = []

    for day in daterange(start, end):
        if day.weekday() >= 5:
            continue

        try:
            mon = day.strftime("%b").upper()
            url = NSE_URL.format(
                year=day.year,
                mon=mon,
                dd=day.strftime("%d"),
            )

            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                continue

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                df = pd.read_csv(zf.open(zf.namelist()[0]))

            df.columns = [c.upper() for c in df.columns]

            # ✅ Equity + NIFTY 50 filter
            df = df[
                (df["SERIES"] == "EQ") &
                (df["SYMBOL"].isin(NIFTY50))
            ]

            if df.empty:
                continue

            df = df.rename(columns={
                "SYMBOL": "ticker",
                "OPEN": "open",
                "HIGH": "high",
                "LOW": "low",
                "CLOSE": "close",
                "TOTTRDQTY": "volume",
            })

            df["ts"] = pd.to_datetime(day)
            df["ts_utc"] = pd.to_datetime(df["ts"], utc=True)
            df["dt"] = df["ts_utc"].dt.date

            frames.append(df[[
                "ticker","ts","open","high","low","close","volume","ts_utc","dt"
            ]])

        except Exception as e:
            log.warning("Bhavcopy failed for %s: %s", day, e)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


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
