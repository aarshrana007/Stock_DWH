from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from .utils.io import read_json, write_json


def _to_iso(ts):
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    return str(ts)


@dataclass
class Watermarks:
    news_last_seen_ts: str | None = None
    market_last_seen_ts: str | None = None

    @staticmethod
    def load(path: Path) -> "Watermarks":
        path = Path(path)

        # First run or missing file
        if not path.exists():
            return Watermarks()

        # File exists but may be corrupted
        try:
            d = read_json(path)
        except json.JSONDecodeError:
            # Treat as first run
            return Watermarks()

        return Watermarks(
            news_last_seen_ts=d.get("news_last_seen_ts"),
            market_last_seen_ts=d.get("market_last_seen_ts"),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        write_json(
            {
                "news_last_seen_ts": _to_iso(self.news_last_seen_ts),
                "market_last_seen_ts": _to_iso(self.market_last_seen_ts),
            },
            path,
        )
