from __future__ import annotations

import argparse
import pandas as pd

from .config import get_paths, get_sources
from .ingest import load_market, load_news
from .silver import build_silver
from .infer import run_inference


# ------------------------------------------------------------------
# SAFE PATH JOIN (LOCAL + S3)
# ------------------------------------------------------------------
def _join(base, *parts):
    if isinstance(base, str):
        return "/".join([base, *parts])
    return base.joinpath(*parts)


# ------------------------------------------------------------------
# WRITE PARTITIONED DATA
# ------------------------------------------------------------------
def _write_partitioned(df: pd.DataFrame, base_path, dt_col="dt"):
    for dt, g in df.groupby(dt_col):
        out_path = _join(base_path, f"{dt_col}={dt}", "part.parquet")
        g.drop(columns=[dt_col]).to_parquet(out_path, index=False)


# ------------------------------------------------------------------
# INGEST
# ------------------------------------------------------------------
def ingest():
    paths = get_paths()
    sources = get_sources()

    raw_prices = load_market(sources)
    raw_news = load_news(sources)

    _write_partitioned(
        raw_prices,
        _join(paths.bronze, "raw_prices")
    )

    _write_partitioned(
        raw_news,
        _join(paths.bronze, "raw_news")
    )


# ------------------------------------------------------------------
# SILVER
# ------------------------------------------------------------------
def silver():
    paths = get_paths()
    build_silver(paths)


# ------------------------------------------------------------------
# INFER
# ------------------------------------------------------------------
def infer():
    paths = get_paths()
    run_inference(paths)


# ------------------------------------------------------------------
# CLI ENTRY
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["ingest", "silver", "infer"],
        help="Pipeline step to run",
    )
    args = parser.parse_args()

    if args.command == "ingest":
        ingest()
    elif args.command == "silver":
        silver()
    elif args.command == "infer":
        infer()


if __name__ == "__main__":
    main()
