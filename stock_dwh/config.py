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
    warehouse: str | Path
    artifacts: str | Path
    logs: Path

    @property
    def bronze(self):
        return f"{self.warehouse}/bronze"

    @property
    def silver(self):
        return f"{self.warehouse}/silver"

    @property
    def gold(self):
        return f"{self.warehouse}/gold"


def get_paths(repo_root: str | Path | None = None) -> Paths:
    root = (
        Path(repo_root)
        if repo_root
        else Path(_env("STOCK_DWH_ROOT", ".")).resolve()
    )

    warehouse_env = _env("STOCK_DWH_WAREHOUSE", str(root / "warehouse"))
    artifacts_env = _env("STOCK_DWH_ARTIFACTS", str(root / "artifacts"))
    logs_env = _env("STOCK_DWH_LOGS", str(root / "logs"))

    # 🚨 CRITICAL: S3 paths MUST remain strings
    warehouse = (
        warehouse_env
        if warehouse_env.startswith("s3://")
        else Path(warehouse_env).resolve()
    )

    artifacts = (
        artifacts_env
        if artifacts_env.startswith("s3://")
        else Path(artifacts_env).resolve()
    )

    logs = Path(logs_env).resolve()
    logs.mkdir(parents=True, exist_ok=True)

    return Paths(
        repo_root=root,
        warehouse=warehouse,
        artifacts=artifacts,
        logs=logs,
    )


# -------------------------------------------------
# Sources
# -------------------------------------------------
@dataclass(frozen=True)
class Sources:
    news_github_csv_path: str
    news_rss_urls: tuple[str, ...]
    market_source: str
    market_csv_path: str


def get_sources() -> Sources:
    return Sources(
        news_github_csv_path=_env(
            "NEWS_GITHUB_CSV_PATH",
            "stock_dwh/Data/news/historical_news.csv",
        ),
        news_rss_urls=tuple(
            filter(None, _env("NEWS_RSS_URLS", "").split(","))
        ),
        market_source=_env("MARKET_SOURCE", "csv"),
        market_csv_path=_env(
            "MARKET_CSV_PATH",
            "stock_dwh/Data/ohlcv.csv",
        ),
    )


# -------------------------------------------------
# Run config
# -------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    timezone: str
    tickers_path: str
    watermark_path: str
    asof_lag_minutes: int


def get_run_config() -> RunConfig:
    return RunConfig(
        timezone=_env("STOCK_DWH_TZ", "Asia/Kolkata"),
        tickers_path=_env(
            "TICKERS_PATH",
            "stock_dwh/Data/market/nifty50.csv",
        ),
        watermark_path=_env(
            "WATERMARK_PATH",
            "_meta/watermarks.json",
        ),
        asof_lag_minutes=int(_env("ASOF_LAG_MINUTES", "5")),
    )
