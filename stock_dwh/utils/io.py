from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _is_s3(path: str | Path) -> bool:
    return str(path).startswith("s3://")


def _get_fs():
    return s3fs.S3FileSystem(
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
        },
    )


# ---------------------------------------------------------
# PARQUET WRITE (schema-safe)
# ---------------------------------------------------------
def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    if df.empty:
        return

    # 🔒 normalize dt to plain string
    if "dt" in df.columns:
        df["dt"] = df["dt"].astype(str)

    table = pa.Table.from_pandas(df, preserve_index=False)

    if _is_s3(path):
        fs = _get_fs()
        with fs.open(path, "wb") as f:
            pq.write_table(
                table,
                f,
                compression="snappy",
                use_dictionary=False,   # 🔥 critical
            )
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table,
            path,
            compression="snappy",
            use_dictionary=False,     # 🔥 critical
        )


# ---------------------------------------------------------
# PARQUET READ (schema-safe merge)
# ---------------------------------------------------------
# def read_parquet(path: str | Path) -> pd.DataFrame:
#     if _is_s3(path):
#         fs = _get_fs()
#         files = fs.find(str(path))
#     else:
#         path = Path(path)
#         if path.is_file():
#             files = [str(path)]
#         else:
#             files = [str(p) for p in path.rglob("*.parquet")]

#     if not files:
#         return pd.DataFrame()

#     dfs = []
#     for f in files:
#         df = pd.read_parquet(f)

#         # 🔒 normalize dt again
#         if "dt" in df.columns:
#             df["dt"] = df["dt"].astype(str)

#         dfs.append(df)

#     return pd.concat(dfs, ignore_index=True)

def read_parquet(path: str | Path) -> pd.DataFrame:
    if _is_s3(path):
        fs = _get_fs()
        files = fs.find(str(path))
        if not files:
            return pd.DataFrame()

        dfs = []
        for f in files:
            df = pd.read_parquet(f, filesystem=fs)

            # normalize dt again
            if "dt" in df.columns:
                df["dt"] = df["dt"].astype(str)

            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    else:
        path = Path(path)
        if path.is_file():
            files = [path]
        else:
            files = list(path.rglob("*.parquet"))

        if not files:
            return pd.DataFrame()

        dfs = []
        for f in files:
            df = pd.read_parquet(f)

            if "dt" in df.columns:
                df["dt"] = df["dt"].astype(str)

            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)



# ---------------------------------------------------------
# JSON READ / WRITE (RESTORED)
# ---------------------------------------------------------
def write_json(obj: dict, path: str | Path) -> None:
    if _is_s3(path):
        fs = _get_fs()
        with fs.open(path, "w") as f:
            json.dump(obj, f, indent=2)
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)


def read_json(path: str | Path) -> dict:
    if _is_s3(path):
        fs = _get_fs()
        with fs.open(path, "r") as f:
            return json.load(f)
    else:
        with open(path, "r") as f:
            return json.load(f)
