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
# PARQUET WRITE (SCHEMA-SAFE, S3-SAFE)
# ---------------------------------------------------------
def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """
    Writes parquet to local or S3 with a stable Arrow schema.
    Fixes dt string vs dictionary encoding issues permanently.
    """
    if df.empty:
        return

    # ---- FORCE dt to pure Python string ----
    if "dt" in df.columns:
        df["dt"] = df["dt"].map(lambda x: str(x) if x is not None else None)

    # ---- BUILD EXPLICIT ARROW SCHEMA ----
    fields = []
    for col in df.columns:
        if col == "dt":
            fields.append(pa.field("dt", pa.string()))
        else:
            try:
                fields.append(pa.field(col, pa.from_numpy_dtype(df[col].dtype)))
            except Exception:
                # fallback for object columns
                fields.append(pa.field(col, pa.string()))

    schema = pa.schema(fields)

    table = pa.Table.from_pandas(
        df,
        schema=schema,
        preserve_index=False,
        safe=False,
    )

    if _is_s3(path):
        fs = _get_fs()
        with fs.open(str(path), "wb") as f:
            pq.write_table(
                table,
                f,
                compression="snappy",
                use_dictionary=False,   # 🔥 critical
                flavor="spark",        # 🔥 avoids Arrow merge quirks
            )
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table,
            path,
            compression="snappy",
            use_dictionary=False,
            flavor="spark",
        )


# ---------------------------------------------------------
# PARQUET READ (S3 + LOCAL, MERGE-SAFE)
# ---------------------------------------------------------
def read_parquet(path: str | Path) -> pd.DataFrame:
    """
    Reads parquet from local or S3, merging partitions safely.
    """
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
# JSON READ / WRITE (LOCAL + S3)
# ---------------------------------------------------------
def write_json(obj: dict, path: str | Path) -> None:
    if _is_s3(path):
        fs = _get_fs()
        with fs.open(str(path), "w") as f:
            json.dump(obj, f, indent=2)
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)


def read_json(path: str | Path) -> dict:
    if _is_s3(path):
        fs = _get_fs()
        with fs.open(str(path), "r") as f:
            return json.load(f)
    else:
        with open(path, "r") as f:
            return json.load(f)
