from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


# --------------------------------------------------
# PATH CONFIG
# --------------------------------------------------
@dataclass(frozen=True)
class Paths:
    repo_root: Path
    warehouse: Path
    artifacts: Path
    logs: Path

    @property
    def bronze(self) -> Path:
        return self.warehouse / "bronze"

    @property
    def silver(self) -> Path:
        return self.warehouse / "silver"

    @property
    def gold(self) -> Path:
        return self.warehouse / "gold"


def get_paths(repo_root: str | Path | None = None) -> Paths:
    root = Path(repo_root) if repo_root else Path(_env("STOCK_DWH_ROOT", ".")).resolve()

    warehouse = Path(
        _env("STOCK_DWH_WAREHOUSE", str(root / "warehouse"))
    ).resolve()

    artifacts = Path(
        _env("STOCK_DWH_ARTIFACTS", str(root / "artifacts"))
    ).resolve()

    logs = Path(
        _env("STOCK_DWH_LOGS", str(root / "logs"))
    ).resolve()

    for p in (warehouse, artifacts, logs):
        p.mkdir(parents=True, exist_ok=True)

    return Paths(
        repo_root=root,
        warehouse=warehouse,
        artifacts=artifacts,
        logs=logs,
    )


# --------------------------------------------------
# SOURCE CONFIG
# --------------------------------------------------
@dataclass(frozen=True)
class Sources:
    # News
    news_github_csv_path: Path
    news_rss_urls: tuple[str, ...]

    # Market
    market_source: str
    market_csv_path: Path


def get_sources() -> Sources:
    return Sources(
        news_github_csv_path=Path(
            _env("NEWS_GITHUB_CSV_PATH", "data/news_data/historical_news.csv")
        ),
        news_rss_urls=tuple(
            u for u in _env("NEWS_RSS_URLS", "").split(",") if u
        ),
        market_source=_env("MARKET_SOURCE", "csv"),
        market_csv_path=Path(
            _env("MARKET_CSV_PATH", "stock_dwh/Data/ohlcv.csv")
        ),
    )


# --------------------------------------------------
# RUNTIME CONFIG
# --------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    timezone: str
    tickers_path: Path
    watermark_path: Path
    asof_lag_minutes: int


def get_run_config() -> RunConfig:
    return RunConfig(
        timezone=_env("STOCK_DWH_TZ", "Asia/Kolkata"),
        tickers_path=Path(
            _env("TICKERS_PATH", "stock_dwh/Data/market/nifty50.csv")
        ),
        watermark_path=Path(
            _env("WATERMARK_PATH", "warehouse/_meta/watermarks.json")
        ),
        asof_lag_minutes=int(_env("ASOF_LAG_MINUTES", "5")),
    )
