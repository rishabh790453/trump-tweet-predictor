"""
Live news headline collection.

Two things this fixes versus dashboard/news_etl.py:

1. Dead feeds. Of the original 12, four fail outright (Reuters and AP resolve or
   handshake-fail, Politico and BBC return malformed XML). They are removed.

2. Stale feeds, which are more dangerous than dead ones. The CNN politics feed
   still returns 30 well-formed entries whose newest item is from December 2022.
   A dead feed is obvious; a stale one silently feeds three-year-old headlines
   into the prediction window and looks completely normal. Every feed is now
   checked for recency and dropped for the cycle if its newest entry is older
   than MAX_FEED_AGE_HOURS.
"""

import hashlib
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser

from .store import utcnow

# Verified reachable and fresh as of 2026-09-13. Feeds are re-validated at
# runtime anyway, so a source going stale degrades the cycle instead of poisoning it.
FEEDS: list[tuple[str, str]] = [
    ("NYT",         "https://rss.nytimes.com/services/xml/rss/nyt/US.xml"),
    ("NYT-Business","https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
    ("Guardian",    "https://www.theguardian.com/us-news/rss"),
    ("CNBC",        "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC-Econ",   "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
    ("Fox News",    "https://moxie.foxnews.com/google-publisher/politics.xml"),
    ("WashPost",    "https://feeds.washingtonpost.com/rss/politics"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("Axios",       "https://api.axios.com/feed/"),
    ("Hill",        "https://thehill.com/news/feed/"),
]

# Feeds confirmed broken on 2026-09-13. Kept documented rather than deleted so
# nobody re-adds them without checking; revive only after verifying recency.
KNOWN_BROKEN = {
    "Reuters":  "feeds.reuters.com no longer resolves (DNS)",
    "AP News":  "TLS handshake fails (UNEXPECTED_EOF_WHILE_READING)",
    "Politico": "malformed XML (not well-formed, invalid token)",
    "BBC":      "malformed XML (mismatched tag)",
    "CNN":      "STALE — parses fine, newest entry Dec 2022",
}

MAX_FEED_AGE_HOURS = 72.0

# Terms that make a headline plausible Trump-reaction material AND market
# relevant. Weighted: a headline naming him directly matters more than one that
# merely mentions the economy.
KEYWORD_WEIGHTS: dict[str, float] = {
    "trump": 3.0, "white house": 2.0, "president": 1.5,
    "tariff": 2.5, "trade war": 2.5, "trade deal": 2.0,
    "fed": 2.0, "federal reserve": 2.5, "powell": 2.5, "interest rate": 2.0,
    "inflation": 1.5, "recession": 1.5, "economy": 1.0, "jobs": 1.0,
    "stock": 1.5, "market": 1.0, "dow": 1.5, "s&p": 1.5, "nasdaq": 1.5,
    "wall street": 1.5, "oil": 1.2, "crypto": 1.5, "bitcoin": 1.5,
    "china": 2.0, "russia": 1.8, "iran": 1.8, "israel": 1.8, "ukraine": 1.8,
    "nato": 1.5, "tax": 1.5, "immigration": 1.5, "border": 1.5,
    "congress": 1.2, "senate": 1.2, "supreme court": 1.5, "judge": 1.2,
    "democrat": 1.2, "republican": 1.0, "election": 1.5, "musk": 1.5,
    "sanctions": 1.5, "ceasefire": 1.5, "tariffs": 2.5, "deal": 0.8,
}

_KW_RE = {kw: re.compile(rf"\b{re.escape(kw)}", re.I) for kw in KEYWORD_WEIGHTS}

# Normalising weight: a headline hitting ~3 strong terms scores ≈1.0.
_RELEVANCE_NORM = 6.0


@dataclass
class NewsItem:
    id: str
    title: str
    source: str
    url: str
    published_at: str
    collected_at: str
    relevance: float

    def as_row(self) -> dict:
        return asdict(self)


def _hash_title(title: str) -> str:
    """
    Hash the normalised title so trivial punctuation/casing edits by a publisher
    do not create a duplicate row for the same story.
    """
    norm = re.sub(r"[^a-z0-9 ]", "", title.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def relevance(title: str) -> float:
    """
    Weighted keyword score in [0, 1].

    The previous implementation computed a word set it never used and then did
    substring counting, so "market" inside "supermarket" counted and multi-word
    terms were unreachable. This matches on word boundaries and weights terms.
    """
    score = sum(w for kw, w in KEYWORD_WEIGHTS.items() if _KW_RE[kw].search(title))
    return round(min(score / _RELEVANCE_NORM, 1.0), 4)


def _parse_entry_date(entry) -> datetime | None:
    for key in ("published", "updated", "created"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def fetch_feed(source: str, url: str, max_items: int = 15) -> tuple[list[NewsItem], str]:
    """
    Fetch one feed. Returns (items, status). Status is 'ok', 'stale', 'empty' or
    'error:...' so the caller can log why a source contributed nothing.
    """
    try:
        parsed = feedparser.parse(url)
    except Exception as exc:
        return [], f"error:{type(exc).__name__}"

    if not parsed.entries:
        detail = str(parsed.get("bozo_exception", "no entries"))[:60]
        return [], f"empty:{detail}"

    now = datetime.now(timezone.utc)
    dates = [d for d in (_parse_entry_date(e) for e in parsed.entries) if d]
    if dates:
        newest = max(dates)
        age_h = (now - newest).total_seconds() / 3600.0
        if age_h > MAX_FEED_AGE_HOURS:
            # The CNN failure mode: parses cleanly, looks healthy, three years old.
            return [], f"stale:{age_h / 24:.0f}d"

    collected = utcnow()
    items: list[NewsItem] = []
    for entry in parsed.entries[:max_items]:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        pub = _parse_entry_date(entry)
        if pub and (now - pub) > timedelta(hours=MAX_FEED_AGE_HOURS):
            continue  # individual stale item inside an otherwise fresh feed
        items.append(NewsItem(
            id=_hash_title(title),
            title=title,
            source=source,
            url=entry.get("link", ""),
            published_at=pub.isoformat() if pub else "",
            collected_at=collected,
            relevance=relevance(title),
        ))
    return items, "ok"


def fetch_all(max_per_feed: int = 15, verbose: bool = True) -> tuple[list[dict], dict[str, str]]:
    """
    Fetch every configured feed. Returns (deduped rows, per-source status).
    Dedup keeps the highest-relevance copy of a story syndicated across sources.
    """
    best: dict[str, NewsItem] = {}
    status: dict[str, str] = {}

    for source, url in FEEDS:
        items, st = fetch_feed(source, url, max_per_feed)
        status[source] = f"{st} ({len(items)})"
        if verbose:
            marker = "ok " if st == "ok" else "SKIP"
            print(f"  [{marker}] {source:<13} {len(items):>3} items  {st}")
        for it in items:
            prev = best.get(it.id)
            if prev is None or it.relevance > prev.relevance:
                best[it.id] = it

    rows = sorted(best.values(), key=lambda x: x.relevance, reverse=True)
    return [r.as_row() for r in rows], status


if __name__ == "__main__":
    print("Fetching live news feeds...\n")
    rows, status = fetch_all()
    ok = sum(1 for s in status.values() if s.startswith("ok"))
    print(f"\n{len(rows)} unique headlines from {ok}/{len(FEEDS)} healthy feeds\n")
    print("Top 12 by relevance:")
    for r in rows[:12]:
        print(f"  {r['relevance']:.2f}  [{r['source']:<12}] {r['title'][:76]}")
