"""
Near-real-time collector.

One process, and only one, owns Truth Social access. That is a hard requirement,
not a style choice: the API enforces a single shared rate-limit budget per
client, and during testing a background backfill running alongside a poller
caused 11 of 12 polls to return HTTP 429. Two "independent" collectors do not
collect twice as fast — they starve each other. `backfill_posts.py` and this
watcher must never run at the same time.

Cadences are deliberately different per source:

  * Truth Social — polled fast, because detecting a post promptly is the whole
    point. One page (20 posts) is enough: he would have to post 20 times inside
    one interval to overflow it.
  * News RSS — polled slowly. Feeds update on the order of minutes and hitting
    ten publishers every 30 seconds is rude and pointless.
  * Predictions — generated on a slower beat again, since a prediction is only
    meaningful once the news window has actually changed.

Everything lands in the same SQLite store as run_cycle.py, so the dashboard,
scoring and Kalshi forecast all read one source of truth.
"""

import argparse
import signal
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone

from . import news_feed, predict, truth_feed
from . import signal as sig
from .store import (connect, finish_run, init_db, insert_news, insert_posts,
                    insert_prediction, latest_post_id, recent_news, start_run,
                    utcnow)

# Measured against the live API. 30s was the fastest spacing that stayed clean
# in testing; going faster buys little, since the median gap between his posts is
# far longer than a minute.
DEFAULT_POST_INTERVAL = 30
DEFAULT_NEWS_INTERVAL = 600        # 10 min
DEFAULT_PREDICT_INTERVAL = 1800    # 30 min

# Consecutive 429s before backing off hard. The API sends no Retry-After, so the
# watcher has to infer its own ceiling.
RATE_LIMIT_BACKOFF = [60, 120, 300, 600]


class Watcher:
    def __init__(self, post_interval=DEFAULT_POST_INTERVAL,
                 news_interval=DEFAULT_NEWS_INTERVAL,
                 predict_interval=DEFAULT_PREDICT_INTERVAL,
                 models=("markov", "ngram", "rag_retrieve"),
                 on_event=None, verbose=True):
        self.post_interval = post_interval
        self.news_interval = news_interval
        self.predict_interval = predict_interval
        self.models = list(models)
        self.on_event = on_event or (lambda kind, payload: None)
        self.verbose = verbose

        self._stop = threading.Event()
        self._consecutive_429 = 0
        self.stats = {"polls": 0, "posts": 0, "news": 0, "preds": 0,
                      "rate_limited": 0, "errors": 0,
                      "started_at": utcnow(), "last_post_at": None}

    # ── logging ──────────────────────────────────────────────────────────────

    def log(self, msg: str):
        if self.verbose:
            print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)

    # ── Truth Social ─────────────────────────────────────────────────────────

    def poll_posts(self) -> int:
        """One page. Returns count of genuinely new posts."""
        self.stats["polls"] += 1
        try:
            with connect() as conn:
                known = latest_post_id(conn)
            # verbose=False: the watcher does its own rate-limit accounting and
            # should not also print the fetcher's retry chatter every 30s.
            page = truth_feed.fetch_page(verbose=False, retries=0)
            self._consecutive_429 = 0
        except Exception as exc:
            if "429" in str(exc):
                self.stats["rate_limited"] += 1
                self._consecutive_429 += 1
                wait = RATE_LIMIT_BACKOFF[
                    min(self._consecutive_429 - 1, len(RATE_LIMIT_BACKOFF) - 1)]
                self.log(f"rate limited ({self._consecutive_429}x) — pausing {wait}s")
                self._stop.wait(wait)
            else:
                self.stats["errors"] += 1
                self.log(f"post poll error: {type(exc).__name__}: {exc}")
            return 0

        rows = []
        for p in page:
            pid = str(p["id"])
            if known and pid == known:
                break
            row = truth_feed._to_row(p)
            row.update(sig.extract(row["text"]).as_row())
            rows.append(row)

        if not rows:
            return 0

        with connect() as conn:
            added = insert_posts(conn, rows)

        if added:
            self.stats["posts"] += added
            self.stats["last_post_at"] = utcnow()
            newest = rows[0]
            preview = (newest["text"] or "(media only)")[:70]
            self.log(f"NEW POST x{added}: {preview}")
            self.on_event("post", {"added": added, "posts": rows[:5]})
        return added

    # ── News ─────────────────────────────────────────────────────────────────

    def poll_news(self) -> int:
        try:
            rows, status = news_feed.fetch_all(verbose=False)
        except Exception as exc:
            self.stats["errors"] += 1
            self.log(f"news poll error: {type(exc).__name__}: {exc}")
            return 0
        with connect() as conn:
            added = insert_news(conn, rows)
        if added:
            self.stats["news"] += added
            healthy = sum(1 for s in status.values() if s.startswith("ok"))
            self.log(f"news: {added} new ({healthy}/{len(news_feed.FEEDS)} feeds ok)")
            self.on_event("news", {"added": added, "healthy": healthy})
        return added

    # ── Predictions ──────────────────────────────────────────────────────────

    def make_predictions(self) -> int:
        run_id = f"watch-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        since = (datetime.now(timezone.utc)
                 - timedelta(hours=predict.DEFAULT_NEWS_WINDOW_HOURS)).isoformat()
        with connect() as conn:
            start_run(conn, run_id)
            headlines = predict.select_headlines(recent_news(conn, since))

        if not headlines:
            with connect() as conn:
                finish_run(conn, run_id, status="ok")
            return 0

        added = 0
        for model in self.models:
            try:
                p = predict.make_prediction(model, headlines)
                with connect() as conn:
                    insert_prediction(conn, p.as_row(run_id))
                added += 1
                self.log(f"predict [{model}]: {(p.text or '(empty)')[:60]}")
                self.on_event("prediction", {
                    "model": model, "text": p.text,
                    "topic": p.signal.topic, "polarity": p.signal.polarity})
            except Exception as exc:
                # One model failing must not lose the others or stop the watcher.
                self.stats["errors"] += 1
                self.log(f"predict [{model}] FAILED: {type(exc).__name__}: {exc}")

        self.stats["preds"] += added
        with connect() as conn:
            finish_run(conn, run_id, status="ok", preds_added=added)
        return added

    # ── Loop ─────────────────────────────────────────────────────────────────

    def run(self):
        init_db()
        self.log(f"watcher up — posts/{self.post_interval}s "
                 f"news/{self.news_interval}s predict/{self.predict_interval}s")
        self.log("NOTE: do not run backfill_posts.py while this is running "
                 "— they share one rate-limit budget")

        next_news = next_predict = 0.0
        while not self._stop.is_set():
            cycle_start = time.monotonic()
            try:
                self.poll_posts()
                now = time.monotonic()
                if now >= next_news:
                    self.poll_news()
                    next_news = now + self.news_interval
                if now >= next_predict:
                    self.make_predictions()
                    next_predict = time.monotonic() + self.predict_interval
            except Exception:
                self.stats["errors"] += 1
                traceback.print_exc()

            # Sleep the remainder of the interval so a slow poll does not push
            # the schedule out indefinitely.
            elapsed = time.monotonic() - cycle_start
            self._stop.wait(max(self.post_interval - elapsed, 1.0))

        self.log("watcher stopped")

    def stop(self):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--post-interval", type=int, default=DEFAULT_POST_INTERVAL)
    ap.add_argument("--news-interval", type=int, default=DEFAULT_NEWS_INTERVAL)
    ap.add_argument("--predict-interval", type=int, default=DEFAULT_PREDICT_INTERVAL)
    ap.add_argument("--models", default="markov,ngram,rag_retrieve")
    args = ap.parse_args()

    w = Watcher(args.post_interval, args.news_interval, args.predict_interval,
                models=[m.strip() for m in args.models.split(",") if m.strip()])

    def handle(signum, frame):
        print("\nstopping...")
        w.stop()

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
    w.run()
    print(f"\nfinal: {w.stats}")


if __name__ == "__main__":
    sys.exit(main() or 0)
