from __future__ import annotations
from datetime import datetime, timezone, timedelta
import re

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def safe_parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    # try ISO first
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        pass
    # common RFC822-ish like "Tue, 10 Dec 2024 10:42:00 GMT"
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s)
    except Exception:
        return None

def dt_to_partition(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    d = dt.astimezone(timezone.utc).date()
    return d.isoformat()
