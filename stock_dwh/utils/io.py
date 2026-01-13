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

    if _is_s3(path):
        fs = _get_fs()
        with fs.open(str(path), "wb") as f:
            pq.write_table(table, f)
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)


def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads BOTH:
    - single parquet file
    - partitioned parquet directory (dataset)
    """

    if _is_s3(path):
        fs = _get_fs()

        # DIRECTORY (partitioned dataset)
        if fs.isdir(str(path)):
            dataset = pq.ParquetDataset(str(path), filesystem=fs)
            return dataset.read().to_pandas()

        # SINGLE FILE
        with fs.open(str(path), "rb") as f:
            return pq.read_table(f).to_pandas()

    else:
        path = Path(path)

        # DIRECTORY (partitioned dataset)
        if path.is_dir():
            dataset = pq.ParquetDataset(path)
            return dataset.read().to_pandas()

        # SINGLE FILE
        return pq.read_table(path).to_pandas()


# ---------------------------------------------------------
# JSON IO
# ---------------------------------------------------------
def write_json(obj: dict, path: Union[str, Path]) -> None:
    if _is_s3(path):
        fs = _get_fs()
        with fs.open(str(path), "w") as f:
            json.dump(obj, f, indent=2)
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)


def read_json(path: Union[str, Path]) -> dict:
    if _is_s3(path):
        fs = _get_fs()
        with fs.open(str(path), "r") as f:
            return json.load(f)
    else:
        with open(path, "r") as f:
            return json.load(f)
