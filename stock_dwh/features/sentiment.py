from __future__ import annotations
import pandas as pd

def finbert_placeholder(df_news: pd.DataFrame) -> pd.DataFrame:
    """Placeholder sentiment implementation.

    You can later replace this with an actual FinBERT pipeline (transformers).
    Output columns:
      - sent_label: POS/NEG/NEU
      - sent_score: float in [-1,1]
      - sent_conf: float in [0,1]
    """
    if df_news.empty:
        return df_news.assign(sent_label=pd.Series(dtype="string"),
                              sent_score=pd.Series(dtype="float"),
                              sent_conf=pd.Series(dtype="float"))
    out = df_news.copy()
    # naive heuristic placeholder (safe + fast)
    text = out["clean_text"].fillna("").astype(str).str.lower()
    pos = text.str.contains(r"beat|surge|up|growth|record|profit")
    neg = text.str.contains(r"miss|down|fall|drop|loss|probe|fraud|concern")
    out["sent_label"] = "NEU"
    out.loc[pos & ~neg, "sent_label"] = "POS"
    out.loc[neg & ~pos, "sent_label"] = "NEG"
    out["sent_score"] = 0.0
    out.loc[out["sent_label"]=="POS","sent_score"] = 0.6
    out.loc[out["sent_label"]=="NEG","sent_score"] = -0.6
    out["sent_conf"] = 0.55
    out.loc[out["sent_label"]!="NEU","sent_conf"] = 0.70
    return out
