from __future__ import annotations
import pandas as pd
from pathlib import Path
import joblib

def train_placeholder(training_set: pd.DataFrame, target_col: str, out_path: Path) -> dict:
    """Simple baseline model: stores mean target per ticker.
    Replace later with LightGBM for real use.
    """
    if training_set.empty:
        raise ValueError("training_set is empty")

    model = training_set.groupby("ticker")[target_col].mean().to_dict()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    return {"model_type": "mean_by_ticker", "n_rows": int(training_set.shape[0])}
