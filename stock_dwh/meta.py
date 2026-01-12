from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .utils.io import read_json, write_json


@dataclass
class Watermarks:
    news_last_seen_ts: str | None = None
    market_last_seen_ts: str | None = None

    @staticmethod
    def load(path: Path) -> "Watermarks":
        path = Path(path)

        # ✅ FIRST RUN SAFE-GUARD
        if not path.exists():
            return Watermarks()

        d = read_json(path)
        return Watermarks(
            news_last_seen_ts=d.get("news_last_seen_ts"),
            market_last_seen_ts=d.get("market_last_seen_ts"),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        write_json(
            {
                "news_last_seen_ts": self.news_last_seen_ts,
                "market_last_seen_ts": self.market_last_seen_ts,
            },
            path,
        )
