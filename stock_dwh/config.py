"""Central configuration for stock_dwh.

This package is designed to be repo-agnostic:
- Data lives under ./warehouse (Bronze/Silver/Gold)
- Models/artifacts live under ./artifacts
- Source files (e.g., news CSV) can be referenced from any directory via env vars or config.

Supports local filesystem and optional S3 warehouse via env vars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


# Backends
DWH_BACKEND = _env("DWH_BACKEND", "local").lower()
S3_BUCKET = _env("S3_BUCKET", "stock-dwh-nse-ai")


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    warehouse: str | Path
    artifacts: Path
    logs: Path

    def _join(self, *parts: str):
        if isinstance(self.warehouse, str):
            return "/".join([self.warehouse.rstrip("/"), *parts])
        return self.warehouse.joinpath(*parts)

    @property
    def bronze(self):
        return self._join("bronze")

    @property
    def silver(self):
        return self._join("silver")

    @property
    def gold(self):
        return self._join("gold")


def get_paths(repo_root: str | Path | None = None) -> Paths:
    root = Path(repo_root) if repo_root else Path(_env("STOCK_DWH_ROOT", ".")).resolve()

    if DWH_BACKEND == "s3":
        warehouse = f"s3://{S3_BUCKET}"
    else:
        warehouse = Path(_env("STOCK_DWH_WAREHOUSE", str(root / "warehouse"))).resolve()
        warehouse.mkdir(parents=True, exist_ok=True)

    artifacts = Path(_env("STOCK_DWH_ARTIFACTS", str(root / "artifacts"))).resolve()
    logs = Path(_env("STOCK_DWH_LOGS", str(root / "logs"))).resolve()
    for p in (artifacts, logs):
        p.mkdir(parents=True, exist_ok=True)

    return Paths(repo_root=root, warehouse=warehouse, artifacts=artifacts, logs=logs)


@dataclass(frozen=True)
class Sources:
    # News sources
    news_github_csv_path: str  # local path
    news_rss_urls: tuple[str, ...]
    # Market data
    market_source: str  # "csv"
    market_csv_path: str


def get_sources() -> Sources:
    return Sources(
        news_github_csv_path=_env("NEWS_GITHUB_CSV_PATH", "Data/news/historical_news.csv"),
        news_rss_urls=tuple(filter(None, _env("NEWS_RSS_URLS", "").split(","))),
        market_source=_env("MARKET_SOURCE", "csv"),
        market_csv_path=_env("MARKET_CSV_PATH", "Data/ohlcv.csv"),
    )


@dataclass(frozen=True)
class RunConfig:
    timezone: str = "Asia/Kolkata"
    tickers_path: str = "Data/market/nifty50.csv"
    watermark_path: str = "warehouse/_meta/watermarks.json"
    asof_lag_minutes: int = 5


def get_run_config() -> RunConfig:
    return RunConfig(
        timezone=_env("STOCK_DWH_TZ", "Asia/Kolkata"),
        tickers_path=_env("TICKERS_PATH", "Data/market/nifty50.csv"),
        watermark_path=_env("WATERMARK_PATH", "warehouse/_meta/watermarks.json"),
        asof_lag_minutes=int(_env("ASOF_LAG_MINUTES", "5")),
    )
