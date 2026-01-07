from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Union, Iterable, List, Optional

import pandas as pd

try:
    import s3fs  # type: ignore
except Exception:  # pragma: no cover
    s3fs = None

PathLike = Union[str, Path]


def _is_s3(p: PathLike) -> bool:
    return isinstance(p, str) and p.startswith("s3://")


def _s3_parts(s3_url: str) -> tuple[str, str]:
    # s3://bucket/key...
    u = s3_url[5:]
    bucket, _, key = u.partition("/")
    return bucket, key


def _get_s3fs():
    if s3fs is None:
        raise ImportError("s3fs is required for S3 paths. Install: pip install s3fs boto3")
    # Use env vars if present (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN)
    return s3fs.S3FileSystem(anon=False)


def ensure_parent(path: PathLike) -> None:
    if _is_s3(path):
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def exists(path: PathLike) -> bool:
    if _is_s3(path):
        fs = _get_s3fs()
        bucket, key = _s3_parts(str(path))
        return fs.exists(f"{bucket}/{key}")
    return Path(path).exists()


def list_parquet_files(root: PathLike) -> List[PathLike]:
    """Return list of parquet files under root (recursive). Works for local and S3."""
    if _is_s3(root):
        fs = _get_s3fs()
        bucket, key = _s3_parts(str(root))
        prefix = f"{bucket}/{key}".rstrip("/")
        # include both root file and recursive
        files = fs.glob(prefix + "/**/*.parquet")
        return [f"s3://{f}" for f in files]
    p = Path(root)
    if p.is_file() and p.suffix == ".parquet":
        return [p]
    return list(p.rglob("*.parquet"))


def write_parquet(df: pd.DataFrame, path: PathLike) -> None:
    ensure_parent(path)
    if _is_s3(path):
        # pandas + s3fs handles the transport
        df.to_parquet(str(path), index=False)
    else:
        df.to_parquet(Path(path), index=False)


def read_parquet(path: PathLike) -> pd.DataFrame:
    """Read a parquet file OR a directory dataset (all parquet under it)."""
    if _is_s3(path):
        # If it's a directory/prefix, load all parquet under it.
        if str(path).endswith("/") or not str(path).lower().endswith(".parquet"):
            files = list_parquet_files(path)
            if not files:
                return pd.DataFrame()
            return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return pd.read_parquet(str(path))
    p = Path(path)
    if p.is_dir():
        files = list_parquet_files(p)
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(p)


def write_json(obj, path: PathLike) -> None:
    ensure_parent(path)
    if _is_s3(path):
        fs = _get_s3fs()
        bucket, key = _s3_parts(str(path))
        with fs.open(f"{bucket}/{key}", "w") as f:
            json.dump(obj, f, indent=2, default=str)
    else:
        with open(Path(path), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)


def read_json(path: PathLike):
    if not exists(path):
        return {}
    if _is_s3(path):
        fs = _get_s3fs()
        bucket, key = _s3_parts(str(path))
        with fs.open(f"{bucket}/{key}", "r") as f:
            return json.load(f)
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)
