"""
Ground truth: actual Truth Social posts.

This is the only source permitted to write the `posts` table, and store.py
enforces that with a CHECK constraint. The old dashboard/live_tweets_etl.py had
three sources, two of which are unusable here:

  * thetrumparchive.com — now returns HTTP 404, dead.
  * RSS quote extraction — scrapes Trump quotes out of news articles. Those are
    the same feeds used as prediction *input*, so scoring a prediction against
    them would be circular: the model would be judged against the very text that
    prompted it. Excluded deliberately, not by oversight.

Only the Truth Social API remains, and it works well — during testing it returned
posts under an hour old.

A large share of his posts carry no text at all (pure image/video reblogs). Those
are stored with empty text so the record is complete, but `posts_with_text` is
what scoring uses; counting media-only posts as prediction misses would make any
hit rate meaningless.
"""

import html
import re
import time
from datetime import datetime, timezone

import requests

from .store import TRUSTED_POST_SOURCE, utcnow

ACCOUNT_ID = "107780257626128497"   # @realDonaldTrump
API_BASE = "https://truthsocial.com/api/v1"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# The API caps a page at 20 regardless of the limit requested.
PAGE_SIZE = 20

# Measured 2026-09-13: roughly 4 pages in quick succession then HTTP 429. There
# is no published limit and no Retry-After header, so the client backs off
# exponentially and gives up rather than hammering. Incremental cycles fetch 1-2
# pages and never approach this; only bulk backfill does.
REQUEST_PAUSE = 3.0
MAX_RETRIES = 4
BACKOFF_BASE = 20.0   # seconds: 20, 40, 80, 160


def _strip_html(raw: str) -> str:
    """Truth Social returns HTML. Preserve paragraph breaks as spaces."""
    if not raw:
        return ""
    text = re.sub(r"<br\s*/?>", " ", raw, flags=re.I)
    text = re.sub(r"</p>\s*<p>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    # The API inserts a space inside URLs ("https://www. wsj.com"); rejoin so the
    # URL stripper in signal.py can remove them cleanly.
    text = re.sub(r"(https?://\S*?)\s+(\S)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def _to_row(post: dict) -> dict:
    text = _strip_html(post.get("content", ""))
    reblog = post.get("reblog")
    return {
        "id": str(post["id"]),
        "created_at": post.get("created_at", ""),
        "collected_at": utcnow(),
        "text": text,
        "has_media": int(bool(post.get("media_attachments"))),
        "is_reblog": int(bool(reblog)),
        "favourites": post.get("favourites_count"),
        "reblogs": post.get("reblogs_count"),
        "url": post.get("url") or post.get("uri") or "",
        "source": TRUSTED_POST_SOURCE,
        # Signals are filled in by the caller (run_cycle) so that this module
        # stays a pure fetcher with no dependency on the feature code.
        "topic": None, "polarity": None, "intensity": None, "tickers": None,
    }


def fetch_page(max_id: str | None = None, timeout: int = 20,
               retries: int = MAX_RETRIES, verbose: bool = True) -> list[dict]:
    """
    One page of statuses, newest first. `max_id` pages backwards in time.

    Retries on 429 with exponential backoff, honouring Retry-After if the server
    ever starts sending it.
    """
    params = {"limit": PAGE_SIZE, "exclude_replies": "true"}
    if max_id:
        params["max_id"] = max_id

    for attempt in range(retries + 1):
        resp = requests.get(
            f"{API_BASE}/accounts/{ACCOUNT_ID}/statuses",
            headers=HEADERS, params=params, timeout=timeout,
        )
        if resp.status_code == 429:
            if attempt == retries:
                resp.raise_for_status()
            wait = float(resp.headers.get("Retry-After") or BACKOFF_BASE * (2 ** attempt))
            if verbose:
                print(f"  [truth] rate limited (429), waiting {wait:.0f}s "
                      f"[attempt {attempt + 1}/{retries}]")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError(f"unexpected API response type: {type(data).__name__}")
        return data
    return []


def fetch_recent(pages: int = 1, stop_at_id: str | None = None,
                 verbose: bool = True) -> list[dict]:
    """
    Fetch up to `pages` pages of posts.

    `stop_at_id` short-circuits once a post we already hold is reached, so an
    incremental cycle costs one request when little has been posted.
    """
    rows: list[dict] = []
    max_id: str | None = None
    seen: set[str] = set()

    for page_no in range(pages):
        try:
            page = fetch_page(max_id, verbose=verbose)
        except requests.HTTPError as exc:
            print(f"  [truth] HTTP error on page {page_no + 1}: {exc}")
            break
        except Exception as exc:
            print(f"  [truth] error on page {page_no + 1}: {type(exc).__name__}: {exc}")
            break

        if not page:
            break

        hit_known = False
        for p in page:
            pid = str(p["id"])
            if pid in seen:
                continue
            seen.add(pid)
            if stop_at_id and pid == stop_at_id:
                hit_known = True
                break
            rows.append(_to_row(p))

        if verbose:
            print(f"  [truth] page {page_no + 1}: {len(page)} posts "
                  f"({sum(1 for r in rows if r['text'])} with text so far)")

        if hit_known or len(page) < PAGE_SIZE:
            break

        max_id = str(page[-1]["id"])
        if page_no < pages - 1:
            time.sleep(REQUEST_PAUSE)

    return rows


def health_check() -> dict:
    """Is the API reachable and is the account returning fresh posts?"""
    try:
        page = fetch_page()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not page:
        return {"ok": False, "error": "empty response"}
    newest = page[0].get("created_at", "")
    try:
        dt = datetime.fromisoformat(newest.replace("Z", "+00:00"))
        age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    except ValueError:
        age_h = float("nan")
    texted = sum(1 for p in page if _strip_html(p.get("content", "")))
    return {
        "ok": True,
        "returned": len(page),
        "with_text": texted,
        "newest_age_hours": round(age_h, 2),
        "newest_at": newest,
    }


if __name__ == "__main__":
    print("Truth Social API health check\n")
    h = health_check()
    for k, v in h.items():
        print(f"  {k:>18}: {v}")

    if h.get("ok"):
        print("\nFetching 2 pages...\n")
        rows = fetch_recent(pages=2)
        texted = [r for r in rows if r["text"]]
        print(f"\n  {len(rows)} posts, {len(texted)} with text "
              f"({len(rows) - len(texted)} media-only)\n")
        for r in texted[:8]:
            print(f"  [{r['created_at'][:16]}] {r['text'][:88]}")
