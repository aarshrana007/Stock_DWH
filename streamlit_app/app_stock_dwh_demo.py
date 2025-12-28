import streamlit as st
import pandas as pd
from pathlib import Path

# This demo app reads GOLD mart written by stock_dwh.
# You can copy the relevant pieces into your existing app.py.

ROOT = Path(".")
GOLD = ROOT / "warehouse" / "gold" / "mart_market_snapshot_topbottom"

st.set_page_config(page_title="AI Market Snapshot (Demo)", layout="wide")

st.title("📈 AI Market Snapshot (NIFTY50 – Demo)")
files = sorted(GOLD.rglob("snapshot.parquet"))
if not files:
    st.info("No snapshot found. Run: `python -m stock_dwh.cli ingest`, `silver`, `infer`")
    st.stop()

df = pd.read_parquet(files[-1])
df["asof_ts"] = pd.to_datetime(df["asof_ts"], utc=True)

st.caption(f"Latest snapshot file: {files[-1]}")
st.dataframe(
    df.sort_values(["bucket","pred"], ascending=[True, False]),
    use_container_width=True,
    hide_index=True,
)
