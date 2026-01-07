"""
Central configuration for stock_dwh.

This package is designed to be repo-agnostic:
- Data lives under ./warehouse (Bronze/Silver/Gold) OR S3
- Models/artifacts live under ./artifacts
- Source files (e.g., news CSV) can be referenced from any directory via env vars or config.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

# -------------------------------------------------------------------
# ENV HELPER
# -------------------------------------------------------------------
def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


# -------------------------------------------------------------------
# BACKEND SWITCH
# -------------------------------------------------------------------
# local | s3
DWH_BACKEND = _env("DWH_BACKEND", "local").lower()

# S3 settings
S3_BUCKET = _env("S3_BUCKET", "stock-dwh-nse-ai")
S3_WAREHOUSE = f"s3://{S3_BUCKET}"


# -------------------------------------------------------------------
# PATHS (LOCAL OR S3)
# -------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    repo_root: Path
    warehouse: str | Path
    artifacts: Path
    logs: Path

    @property
    def bronze(self): return self._join("bronze")
    @property
    def silver(self): return self._join("silver")
    @property
    def gold(self): return self._join("gold")

    def _join(self, *parts):
        if isinstance(self.warehouse, str):
            return "/".join([self.warehouse, *parts])
        return self.warehouse.joinpath(*parts)


def get_paths(repo_root: str | Path | None = None) -> Paths:
    root = Path(repo_root) if repo_root else Path(
        _env("STOCK_DWH_ROOT", ".")
    ).resolve()

    # ---------------------------
    # Warehouse (local or S3)
    # ---------------------------
    if DWH_BACKEND == "s3":
        warehouse = S3_WAREHOUSE
    else:
        warehouse = Path(
            _env("STOCK_DWH_WAREHOUSE", str(root / "warehouse"))
        ).resolve()
        warehouse.mkdir(parents=True, exist_ok=True)

    artifacts = Path(
        _env("STOCK_DWH_ARTIFACTS", str(root / "artifacts"))
    ).resolve()
    logs = Path(
        _env("STOCK_DWH_LOGS", str(root / "logs"))
    ).resolve()

    for p in (artifacts, logs):
        p.mkdir(parents=True, exist_ok=True)

    return Paths(
        repo_root=root,
        warehouse=warehouse,
        artifacts=artifacts,
        logs=logs,
    )


# -------------------------------------------------------------------
# DATA SOURCES
# -------------------------------------------------------------------
@dataclass(frozen=True)
class Sources:
    # News sources
    news_github_csv_path: str
    news_rss_urls: tuple[str, ...]

    # Market data
    market_source: str
    market_csv_path: str


def get_sources() -> Sources:
    return Sources(
        news_github_csv_path=_env(
            "NEWS_GITHUB_CSV_PATH",
            "data/news_data/historical_news.csv"
        ),
        news_rss_urls=tuple(
            filter(None, _env("NEWS_RSS_URLS", "").split(","))
        ),
        market_source=_env("MARKET_SOURCE", "csv"),
        market_csv_path=_env(
            "MARKET_CSV_PATH",
            "stock_dwh/Data/ohlcv.csv"
        ),
    )


# -------------------------------------------------------------------
# RUNTIME CONFIG
# -------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    timezone: str = "Asia/Kolkata"
    tickers_path: str = "data/market/nifty50_tickers.txt"
    watermark_path: str = "warehouse/_meta/watermarks.json"
    asof_lag_minutes: int = 5


def get_run_config() -> RunConfig:
    return RunConfig(
        timezone=_env("STOCK_DWH_TZ", "Asia/Kolkata"),
        tickers_path=_env(
            "TICKERS_PATH",
            "data/market/nifty50_tickers.txt"
        ),
        watermark_path=_env(
            "WATERMARK_PATH",
            "warehouse/_meta/watermarks.json"
        ),
        asof_lag_minutes=int(_env("ASOF_LAG_MINUTES", "5")),
    )
