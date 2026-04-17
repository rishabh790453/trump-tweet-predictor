"""
Live Trump post fetcher — tries multiple sources with fallbacks:
  1. thetrumparchive.com JSON API  (archival, good coverage)
  2. Truth Social Mastodon API     (real-time, sometimes Cloudflare-blocked)
  3. Quote extraction from news RSS (always works, extracts verbatim quotes)
"""

import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Optional

import feedparser
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html,application/xhtml+xml,*/*",
}

TRUMP_TS_ACCOUNT = "107780257626128497"


# ── Source 1: thetrumparchive.com ─────────────────────────────────────────────

def _fetch_trump_archive(n: int = 30) -> list[dict]:
    """thetrumparchive.com has a public JSON API with recent Truth Social posts."""
    try:
        url = f"https://thetrumparchive.com/api?results={n}&onlyTweets=true"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        # API returns list of {id, text, date, isRT, device, favorites, retweets}
        if not isinstance(data, list):
            data = data.get("results", data.get("data", []))
        out = []
        for item in data[:n]:
            text = item.get("text", "").strip()
            if not text or item.get("isRT"):
                continue
            ts = item.get("date", "")
            out.append({
                "id": hashlib.md5(text.encode()).hexdigest()[:10],
                "content": text[:280],
                "ts": ts,
                "platform": "Truth Social",
                "likes": str(item.get("favorites", "")),
                "source": "thetrumparchive.com",
                "sp500": None,
                "dow": None,
            })
        return out
    except Exception as e:
        print(f"[LiveTweets] trump archive error: {e}")
        return []


# ── Source 2: Truth Social Mastodon API ──────────────────────────────────────

def _fetch_truth_social(n: int = 20) -> list[dict]:
    """Direct Truth Social API — works when not rate-limited by Cloudflare."""
    try:
        url = (
            f"https://truthsocial.com/api/v1/accounts/"
            f"{TRUMP_TS_ACCOUNT}/statuses?limit={n}&exclude_replies=true"
        )
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"[LiveTweets] Truth Social API: HTTP {resp.status_code}")
            return []
        posts = resp.json()
        if not isinstance(posts, list):
            return []
        out = []
        for p in posts:
            # Strip HTML tags from content
            raw = p.get("content", "")
            text = re.sub(r"<[^>]+>", " ", raw).strip()
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            out.append({
                "id": hashlib.md5(text.encode()).hexdigest()[:10],
                "content": text[:280],
                "ts": p.get("created_at", ""),
                "platform": "Truth Social",
                "likes": str(p.get("favourites_count", "")),
                "source": "Truth Social",
                "sp500": None,
                "dow": None,
            })
        return out
    except Exception as e:
        print(f"[LiveTweets] Truth Social error: {e}")
        return []


# ── Source 3: Extract Trump quotes from news RSS ──────────────────────────────

_QUOTE_PATTERN = re.compile(
    r'["\u201c\u201d]([^"\u201c\u201d]{40,280})["\u201c\u201d]'
)

_TRUMP_ATTR = re.compile(
    r'\bTrump\b.*?(?:wrote|posted|said|stated|declared|wrote on|posted on|Truth Social)',
    re.IGNORECASE,
)

NEWS_FEEDS_FOR_QUOTES = [
    ("Fox News",  "https://moxie.foxnews.com/google-publisher/politics.xml"),
    ("AP News",   "https://feeds.ap.org/rss/apf-topnews"),
    ("Reuters",   "https://feeds.reuters.com/reuters/topNews"),
    ("CNN",       "http://rss.cnn.com/rss/cnn_allpolitics.rss"),
    ("Politico",  "https://www.politico.com/rss/politics08.xml"),
]


def _extract_quotes_from_news(n: int = 10) -> list[dict]:
    """
    Scan news RSS descriptions for articles where Trump is the speaker.
    Extracts the quoted text as a synthetic 'post'.
    """
    results = []
    seen_texts: set[str] = set()

    for source, url in NEWS_FEEDS_FOR_QUOTES:
        if len(results) >= n:
            break
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                title = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")
                combined = f"{title}. {summary}"

                # Only look at articles clearly about Trump speaking
                if not _TRUMP_ATTR.search(combined):
                    continue

                matches = _QUOTE_PATTERN.findall(combined)
                for match in matches:
                    text = match.strip()
                    if len(text) < 40:
                        continue
                    key = text[:80].lower()
                    if key in seen_texts:
                        continue
                    seen_texts.add(key)
                    pub = entry.get("published", datetime.now(timezone.utc).isoformat())
                    results.append({
                        "id": hashlib.md5(text.encode()).hexdigest()[:10],
                        "content": text[:280],
                        "ts": pub,
                        "platform": "Truth Social",
                        "likes": "",
                        "source": f"via {source}",
                        "sp500": None,
                        "dow": None,
                    })
                    if len(results) >= n:
                        break
        except Exception as e:
            print(f"[LiveTweets] Quote extract error ({source}): {e}")

    return results


# ── Public interface ───────────────────────────────────────────────────────────

def fetch_live_tweets(n: int = 30) -> list[dict]:
    """
    Try sources in order, combine results, deduplicate by content hash.
    Returns up to n posts, newest first.
    """
    all_posts: list[dict] = []
    seen_ids: set[str] = set()

    def _add(posts: list[dict]):
        for p in posts:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                all_posts.append(p)

    # Try archive first (most reliable, ~48h lag sometimes)
    archive = _fetch_trump_archive(n)
    if archive:
        print(f"[LiveTweets] Archive: {len(archive)} posts")
        _add(archive)

    # Try Truth Social direct (real-time but may be blocked)
    if len(all_posts) < 5:
        ts_posts = _fetch_truth_social(n)
        if ts_posts:
            print(f"[LiveTweets] Truth Social: {len(ts_posts)} posts")
            _add(ts_posts)

    # Always augment with news quotes (catches today's statements)
    quotes = _extract_quotes_from_news(10)
    if quotes:
        print(f"[LiveTweets] News quotes: {len(quotes)} extracted")
        _add(quotes)

    if not all_posts:
        print("[LiveTweets] All sources failed — returning empty")

    return all_posts[:n]
