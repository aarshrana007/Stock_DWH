import pandas as pd
import requests
from pathlib import Path

NIFTY50_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"
OUT_PATH = Path("data/market/nifty50.csv")


def sync_nifty50():
    print("🔄 Fetching NIFTY 50 constituents from NSE...")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(NIFTY50_URL, headers=headers, timeout=20)
    r.raise_for_status()

    df = pd.read_csv(pd.compat.StringIO(r.text))

    # NSE column is usually 'Symbol'
    df = df.rename(columns={"Symbol": "ticker"})

    df["ticker"] = df["ticker"].str.upper().str.strip()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["ticker"]].drop_duplicates().to_csv(OUT_PATH, index=False)

    print(f"✅ NIFTY 50 synced: {len(df)} stocks")
    print(f"📁 Saved to: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    sync_nifty50()
