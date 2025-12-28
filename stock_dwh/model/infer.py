from __future__ import annotations
import pandas as pd
from pathlib import Path
import joblib

def load_model(model_path: Path):
    return joblib.load(model_path)

def predict(model, features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["ticker","asof_ts","pred","conf"])
    out = features.copy()
    out["pred"] = out["ticker"].map(lambda t: float(model.get(t, 0.0)))
    # simple confidence proxy
    out["conf"] = (out.get("news_conf_mean_24h", 0.0) + 0.3).clip(0.0, 1.0) if "news_conf_mean_24h" in out.columns else 0.5
    return out[["ticker","asof_ts","pred","conf"]]
