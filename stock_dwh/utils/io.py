from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _is_s3(path: Union[str, Path]) -> bool:
    return str(path).startswith("s3://")


def _get_fs():
    return s3fs.S3FileSystem()


# ---------------------------------------------------------
# Internal: fix tz-aware datetimes
# ---------------------------------------------------------
def _fix_tz(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


# ---------------------------------------------------------
# Parquet IO
# ---------------------------------------------------------
def write_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    if df is None or df.empty:
        return

    df = _fix_tz(df)
    table = pa.Table.from_pandas(df, preserve_index=False)

    # ---- LOCAL FIRST ----
    path_str = str(path)
    if not _is_s3(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        return

    # ---- S3 ----
    fs = _get_fs()
    with fs.open(path_str, "wb") as f:
        pq.write_table(table, f)


def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    path_str = str(path)

    # -------------------------------------------------
    # 1️⃣ LOCAL FILESYSTEM ALWAYS HAS PRIORITY
    # -------------------------------------------------
    local_path = Path(path)
    if local_path.exists():
        # Directory (partitioned parquet)
        if local_path.is_dir():
            return pq.ParquetDataset(local_path).read().to_pandas()

        # Single parquet file
        return pq.read_table(local_path).to_pandas()

    # -------------------------------------------------
    # 2️⃣ ONLY THEN TRY S3 (explicit s3://)
    # -------------------------------------------------
    if _is_s3(path):
        fs = _get_fs()

        # Directory (dataset)
        if fs.isdir(path_str):
            return pq.ParquetDataset(path_str, filesystem=fs).read().to_pandas()

        # Single file
        with fs.open(path_str, "rb") as f:
            return pq.read_table(f).to_pandas()

    # -------------------------------------------------
    # 3️⃣ Nothing found
    # -------------------------------------------------
    raise FileNotFoundError(path_str)


# ---------------------------------------------------------
# JSON IO
# ---------------------------------------------------------
def write_json(obj: dict, path: Union[str, Path]) -> None:
    path_str = str(path)

    if not _is_s3(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        return

    fs = _get_fs()
    with fs.open(path_str, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: Union[str, Path]) -> dict:
    path_str = str(path)

    if not _is_s3(path):
        with open(path_str, "r") as f:
            return json.load(f)

    fs = _get_fs()
    with fs.open(path_str, "r") as f:
        return json.load(f)
