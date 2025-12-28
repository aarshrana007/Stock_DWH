from __future__ import annotations
import hashlib

def sha1_text(text: str) -> str:
    t = (text or "").strip().encode("utf-8", errors="ignore")
    return hashlib.sha1(t).hexdigest()

def canonicalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())
