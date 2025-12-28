from __future__ import annotations
import pandas as pd
from pathlib import Path
from datetime import timezone
from ..utils.time import safe_parse_dt, dt_to_partition

def load_market_csv(csv_path: Path) -> pd.DataFrame:
    """Load OHLCV from a CSV file.

    Expected columns (case-insensitive):
      - ticker OR symbol
      - datetime OR timestamp OR date
      - open, high, low, close, volume
    """
    df = pd.read_csv(csv_path)
    colmap = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    t_c = pick("ticker", "symbol", "stock")
    ts_c = pick("datetime", "timestamp", "date", "time")
    o_c = pick("open", "o")
    h_c = pick("high", "h")
    l_c = pick("low", "l")
    c_c = pick("close", "c")
    v_c = pick("volume", "vol", "v")

    out = pd.DataFrame({
        "ticker": df[t_c] if t_c else "",
        "ts": df[ts_c] if ts_c else "",
        "open": df[o_c] if o_c else None,
        "high": df[h_c] if h_c else None,
        "low": df[l_c] if l_c else None,
        "close": df[c_c] if c_c else None,
        "volume": df[v_c] if v_c else None,
    })
    # parse ts
    pub, dt_part = [], []
    for s in out["ts"].fillna("").astype(str).tolist():
        d = safe_parse_dt(s)
        if d is None:
            pub.append(pd.NaT); dt_part.append(None)
        else:
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            d = d.astimezone(timezone.utc)
            pub.append(d); dt_part.append(dt_to_partition(d))
    out["ts_utc"] = pub
    out["dt"] = dt_part
    out = out.dropna(subset=["ts_utc"]).reset_index(drop=True)
    return out

def filter_incremental(df: pd.DataFrame, last_seen_ts: str | None) -> pd.DataFrame:
    if df.empty or not last_seen_ts:
        return df
    try:
        ts = pd.to_datetime(last_seen_ts, utc=True)
        return df[df["ts_utc"] > ts].reset_index(drop=True)
    except Exception:
        return df

def update_last_seen(df: pd.DataFrame, last_seen_ts: str | None) -> str | None:
    if df.empty:
        return last_seen_ts
    mx = pd.to_datetime(df["ts_utc"], utc=True).max()
    return mx.isoformat() if pd.notna(mx) else last_seen_ts
