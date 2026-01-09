import os
import streamlit as st
import pandas as pd
from pathlib import Path
import s3fs

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"

LOCAL_GOLD = Path("warehouse/gold/mart_market_snapshot_topbottom")
S3_GOLD = os.getenv(
    "S3_GOLD_PATH",
    "s3://stock-dwh-nse-ai/gold/mart_market_snapshot_topbottom"
)

st.set_page_config(
    page_title="AI Market Snapshot (NIFTY50)",
    layout="wide"
)

st.title("📈 AI Market Snapshot (NIFTY 50 – Demo)")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
def load_latest_snapshot():
    if USE_S3:
        fs = s3fs.S3FileSystem(
            key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            client_kwargs={
                "region_name": os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
            },
        )
        files = sorted(fs.find(S3_GOLD))
        files = [f for f in files if f.endswith("snapshot.parquet")]
        if not files:
            return None, None
        latest = files[-1]
        return latest, pd.read_parquet(latest, filesystem=fs)
    else:
        files = sorted(LOCAL_GOLD.rglob("snapshot.parquet"))
        if not files:
            return None, None
        latest = files[-1]
        return latest, pd.read_parquet(latest)


path, df = load_latest_snapshot()

if df is None:
    st.info(
        "No snapshot found.\n\n"
        "Run:\n"
        "`python -m stock_dwh.stock_dwh.cli ingest`\n"
        "`python -m stock_dwh.stock_dwh.cli silver`\n"
        "`python -m stock_dwh.stock_dwh.cli infer`"
    )
    st.stop()

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------
df["asof_ts"] = pd.to_datetime(df["asof_ts"], utc=True)

st.caption(f"Latest snapshot: `{path}`")

st.dataframe(
    df.sort_values(["bucket", "pred"], ascending=[True, False]),
    use_container_width=True,
    hide_index=True,
)

# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------
c1, c2, c3 = st.columns(3)

c1.metric("Total Stocks", len(df))
c2.metric("Top Signals", (df["bucket"] == "TOP").sum())
c3.metric("Bottom Signals", (df["bucket"] == "BOTTOM").sum())
