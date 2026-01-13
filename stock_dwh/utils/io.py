from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Union

import pyarrow as pa
import pyarrow.parquet as pq

# -------------------------------------------------------------------
# Optional S3 helpers (safe even if not used)
# -------------------------------------------------------------------
def _is_s3(path: Union[str, Path]) -> bool:
    return str(path).startswith("s3://")


def _get_fs():
    import pyarrow.fs as fs
    return fs.S3FileSystem()


# -------------------------------------------------------------------
# Core parquet writer (FIXED + HARDENED)
# -------------------------------------------------------------------
def write_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Write a Pandas DataFrame to Parquet with a stable Arrow schema.

    Lakehouse contract enforced:
    - All timestamps must be tz-naive UTC
    - dt column (if present) is STRING (partition-safe)
    - No pyarrow tz crashes
    """

    if df is None or df.empty:
        return

    df = df.copy()

    # ------------------------------------------------------------
    # 1️⃣ Normalize dt partition column (your repo expects string)
    # ------------------------------------------------------------
    if "dt" in df.columns:
        df["dt"] = df["dt"].astype(str)

    # ------------------------------------------------------------
    # 2️⃣ FIX ROOT CAUSE: tz-aware datetime → tz-naive UTC
    # ------------------------------------------------------------
    for col in df.columns:
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = (
                df[col]
                .dt.tz_convert("UTC")
                .dt.tz_localize(None)
            )

    # ------------------------------------------------------------
    # 3️⃣ Build explicit Arrow schema (NO inference surprises)
    # ------------------------------------------------------------
    fields = []
    for col in df.columns:
        series = df[col]

        if col == "dt":
            fields.append(pa.field(col, pa.string()))

        elif pd.api.types.is_integer_dtype(series):
            fields.append(pa.field(col, pa.int64()))

        elif pd.api.types.is_float_dtype(series):
            fields.append(pa.field(col, pa.float64()))

        elif pd.api.types.is_bool_dtype(series):
            fields.append(pa.field(col, pa.bool_()))

        elif pd.api.types.is_datetime64_any_dtype(series):
            fields.append(pa.field(col, pa.timestamp("ns")))

        else:
            fields.append(pa.field(col, pa.string()))

    schema = pa.schema(fields)

    # ------------------------------------------------------------
    # 4️⃣ Convert Pandas → Arrow
    # ------------------------------------------------------------
    table = pa.Table.from_pandas(
        df,
        schema=schema,
        preserve_index=False
    )

    # ------------------------------------------------------------
    # 5️⃣ Write (Local or S3)
    # ------------------------------------------------------------
    if _is_s3(path):
        fs = _get_fs()
        with fs.open_output_stream(str(path)) as f:
            pq.write_table(table, f)
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
