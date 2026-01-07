from __future__ import annotations

import argparse
import pandas as pd
from stock_dwh.config import get_paths, get_sources, get_run_config
from stock_dwh.ingest import load_market, load_news
from stock_dwh.silver import build_silver
from stock_dwh.infer import run_inference


# ------------------------------------------------------------------
# PATH JOIN HELPER (CRITICAL FIX)
# ------------------------------------------------------------------
def _join(base, *parts):
    """
    Join paths safely for both:
    - pathlib.Path (local)
    - str (s3://...)
    """
    if isinstance(base, str):
        return "/".join([base, *parts])
    return base.joinpath(*parts)


# ------------------------------------------------------------------
# WRITE PARTITIONED DATA
# ------------------------------------------------------------------
def _write_partitioned(df: pd.DataFrame, base_path, dt_col="dt"):
    """
    Write DataFrame partitioned by dt column
    """
    for dt, g in df.groupby(dt_col):
        out = _join(base_path, f"{dt_col}={dt}", "part.parquet")
        g.drop(columns=[dt_col]).to_parquet(out, index=False)


# ------------------------------------------------------------------
# INGEST
# ------------------------------------------------------------------
def ingest():
    paths = get_paths()
    sources = get_sources()

    # Load data
    raw_prices = load_market(sources)
    raw_news = load_news(sources)

    # Write bronze
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
# CLI
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
