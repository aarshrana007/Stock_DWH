from __future__ import annotations
import pandas as pd
import feedparser
from pathlib import Path
from datetime import timezone
from ..utils.time import safe_parse_dt, dt_to_partition
from ..utils.dedup import sha1_text, canonicalize_text

REQUIRED_COLS = ("title", "link", "published", "summary", "source")

def load_news_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize known schemas
    colmap = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    title_c = pick("title", "headline")
    link_c = pick("link", "url")
    pub_c = pick("published", "date", "datetime", "time")
    summ_c = pick("summary", "description", "desc")
    src_c  = pick("source", "publisher")

    out = pd.DataFrame({
        "title": df[title_c] if title_c else "",
        "link": df[link_c] if link_c else "",
        "published": df[pub_c] if pub_c else "",
        "summary": df[summ_c] if summ_c else "",
        "source": df[src_c] if src_c else "csv",
    })
    return out

def load_rss(urls: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for url in urls:
        if not url:
            continue
        feed = feedparser.parse(url)
        src = (feed.feed.get("title") or "rss").strip()
        for e in feed.entries:
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            published = e.get("published") or e.get("updated") or ""
            summary = (e.get("summary") or e.get("description") or "").strip()
            rows.append({"title": title, "link": link, "published": published, "summary": summary, "source": src})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(REQUIRED_COLS))

def normalize_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(
            canonical_id=pd.Series(dtype="string"),
            published_ts=pd.Series(dtype="datetime64[ns, UTC]"),
            clean_text=pd.Series(dtype="string"),
            dt=pd.Series(dtype="string"),
        )
    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)
    df["link"] = df["link"].fillna("").astype(str)
    df["source"] = df["source"].fillna("").astype(str)

    df["clean_text"] = (df["title"] + " " + df["summary"]).map(canonicalize_text)
    df["canonical_id"] = df["clean_text"].map(sha1_text)

    pub = []
    dt_part = []
    for s in df["published"].fillna("").astype(str).tolist():
        d = safe_parse_dt(s)
        if d is None:
            pub.append(pd.NaT)
            dt_part.append(None)
        else:
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            pub.append(d.astimezone(timezone.utc))
            dt_part.append(dt_to_partition(d))
    df["published_ts"] = pub
    df["dt"] = dt_part
    # drop rows without date
    df = df.dropna(subset=["published_ts"]).reset_index(drop=True)
    return df

def filter_incremental(df: pd.DataFrame, last_seen_ts: str | None) -> pd.DataFrame:
    if df.empty or not last_seen_ts:
        return df
    try:
        ts = pd.to_datetime(last_seen_ts, utc=True)
        return df[df["published_ts"] > ts].reset_index(drop=True)
    except Exception:
        return df

def update_last_seen(df: pd.DataFrame, last_seen_ts: str | None) -> str | None:
    if df.empty:
        return last_seen_ts
    mx = pd.to_datetime(df["published_ts"], utc=True).max()
    return mx.isoformat() if pd.notna(mx) else last_seen_ts
