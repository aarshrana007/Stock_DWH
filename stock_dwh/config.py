"""Central configuration for stock_dwh.

This package is designed to be repo-agnostic:
- Data lives under ./warehouse (Bronze/Silver/Gold)
- Models/artifacts live under ./artifacts
- Source files (e.g., news CSV) can be referenced from any directory via env vars or config.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default

@dataclass(frozen=True)
class Paths:
    repo_root: Path
    warehouse: Path
    artifacts: Path
    logs: Path

    @property
    def bronze(self) -> Path: return self.warehouse / "bronze"
    @property
    def silver(self) -> Path: return self.warehouse / "silver"
    @property
    def gold(self) -> Path: return self.warehouse / "gold"

def get_paths(repo_root: str | Path | None = None) -> Paths:
    root = Path(repo_root) if repo_root else Path(_env("STOCK_DWH_ROOT", ".")).resolve()
    warehouse = Path(_env("STOCK_DWH_WAREHOUSE", str(root / "warehouse"))).resolve()
    artifacts = Path(_env("STOCK_DWH_ARTIFACTS", str(root / "artifacts"))).resolve()
    logs = Path(_env("STOCK_DWH_LOGS", str(root / "logs"))).resolve()
    for p in (warehouse, artifacts, logs):
        p.mkdir(parents=True, exist_ok=True)
    return Paths(repo_root=root, warehouse=warehouse, artifacts=artifacts, logs=logs)

@dataclass(frozen=True)
class Sources:
    # News sources
    news_github_csv_path: str  # local path after you fetch it or mount it
    news_rss_urls: tuple[str, ...]
    # Market data
    market_source: str  # "nsepy", "yfinance", "csv" etc.
    market_csv_path: str

def get_sources() -> Sources:
    # Keep defaults intentionally "safe": no remote access required.
    return Sources(
        news_github_csv_path=_env("NEWS_GITHUB_CSV_PATH", "data/news_data/historical_news.csv"),
        news_rss_urls=tuple(filter(None, _env("NEWS_RSS_URLS", "").split(","))),
        market_source=_env("MARKET_SOURCE", "csv"),
        market_csv_path=_env("MARKET_CSV_PATH", "stock_dwh/data/market/ohlcv.csv"),
    )

@dataclass(frozen=True)
class RunConfig:
    timezone: str = "Asia/Kolkata"
    # NIFTY50 tickers file (optional). If absent, system will still run for whatever data exists.
    tickers_path: str = "data/market/nifty50_tickers.txt"
    watermark_path: str = "warehouse/_meta/watermarks.json"
    asof_lag_minutes: int = 5  # avoid partial candles/news

def get_run_config() -> RunConfig:
    return RunConfig(
        timezone=_env("STOCK_DWH_TZ", "Asia/Kolkata"),
        tickers_path=_env("TICKERS_PATH", "data/market/nifty50_tickers.txt"),
        watermark_path=_env("WATERMARK_PATH", "warehouse/_meta/watermarks.json"),
        asof_lag_minutes=int(_env("ASOF_LAG_MINUTES", "5")),
    )
