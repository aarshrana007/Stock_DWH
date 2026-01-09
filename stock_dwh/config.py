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
