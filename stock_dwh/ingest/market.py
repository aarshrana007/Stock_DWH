from __future__ import annotations

import pandas as pd
import requests
import zipfile
import io
from datetime import datetime, timedelta
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# --------------------------------------------------
# NIFTY 50 symbols (NSE cash market)
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

NSE_URL = (
    "https://archives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
)

SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# ==================================================
# CLI EXPECTED FUNCTIONS (SIGNATURES MUST MATCH)
# ==================================================

def load_market_csv(mkt_path: str | Path) -> pd.DataFrame:
    """
    CLI entry point.
    CSV path is ignored; NSE Bhavcopy is used instead.
    """
    log.info("Loading market CSV: %s", mkt_path)

    end = datetime.today().date()
    start = end - timedelta(days=365)

    session = requests.Session()
    session.headers.update(SESSION_HEADERS)

    frames = []

    for day in daterange(start, end):
        try:
            url = NSE_URL.format(date=day.strftime("%Y%m%d"))
            resp = session.get(url, timeout=20)

            if resp.status_code != 200:
                continue

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                name = zf.namelist()[0]
                df = pd.read_csv(zf.open(name))

            df.columns = [c.lower() for c in df.columns]

            # Filter EQ series + NIFTY 50
            df = df[
                (df["series"] == "EQ") &
                (df["symbol"].isin(NIFTY50))
            ]

            if df.empty:
                continue

            df = df.rename(columns={
                "symbol": "ticker",
                "open_price": "open",
                "high_price": "high",
                "low_price": "low",
                "close_price": "close",
                "ttl_trd_qnty": "volume",
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


def filter_incremental(
    df: pd.DataFrame,
    last_seen_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    if df.empty or last_seen_ts is None:
        return df
    return df[df["ts_utc"] > last_seen_ts]


def update_last_seen(
    df: pd.DataFrame,
    prev_last_seen_ts: pd.Timestamp | None,
) -> pd.Timestamp | None:
    if df.empty:
        return prev_last_seen_ts
    return df["ts_utc"].max()


# ==================================================
# HELPERS
# ==================================================

def daterange(start, end):
    for n in range((end - start).days + 1):
        yield start + timedelta(n)
