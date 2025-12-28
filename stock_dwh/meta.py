from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from .utils.io import read_json, write_json

@dataclass
class Watermarks:
    news_last_seen_ts: str | None = None
    market_last_seen_ts: str | None = None

    @staticmethod
    def load(path: Path) -> "Watermarks":
        d = read_json(path)
        return Watermarks(
            news_last_seen_ts=d.get("news_last_seen_ts"),
            market_last_seen_ts=d.get("market_last_seen_ts"),
        )

    def save(self, path: Path) -> None:
        write_json(
            {
                "news_last_seen_ts": self.news_last_seen_ts,
                "market_last_seen_ts": self.market_last_seen_ts,
            },
            path,
        )
