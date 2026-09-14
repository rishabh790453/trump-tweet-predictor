#!/usr/bin/env python3
"""
One cycle of the live loop:

    collect news  →  predict  →  collect actual posts  →  match elapsed  →  report

Run one-shot from cron, or with --daemon to stay resident. The daemon matters for
the heavy models: the fine-tuned Llama is 16GB and takes minutes to load on MPS,
so a cron job that loads it every cycle spends most of its life loading. Cheap
models (markov, ngram, rag_retrieve) are fine one-shot.

    # every 30 min, cheap models
    venv_mac/bin/python3 run_cycle.py --models markov,ngram,rag_retrieve

    # resident, keeps the fine-tuned model warm
    venv_mac/bin/python3 run_cycle.py --daemon --interval 1800 \
        --models finetuned_llama,rag_retrieve

Ordering matters: posts are collected BEFORE matching so the horizon that just
elapsed is scored against the freshest ground truth available. Predictions are
made from news collected in the same cycle, never from news collected later.
"""

import argparse
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone

from pipeline import match, news_feed, predict, truth_feed
from pipeline import signal as sig
from pipeline.store import (connect, finish_run, init_db, insert_news,
                            insert_posts, insert_prediction, jdump,
                            latest_post_id, recent_news, start_run, stats, utcnow)


def collect_news(conn, verbose=True) -> tuple[int, list]:
    if verbose:
        print("\n── news " + "─" * 62)
    rows, status = news_feed.fetch_all(verbose=verbose)
    added = insert_news(conn, rows)
    if verbose:
        healthy = sum(1 for s in status.values() if s.startswith("ok"))
        print(f"  {len(rows)} fetched, {added} new  ({healthy}/{len(news_feed.FEEDS)} feeds healthy)")
    return added, rows


def collect_posts(conn, pages=2, verbose=True) -> int:
    """Ground truth. Stops early at the newest post already stored."""
    if verbose:
        print("\n── ground truth " + "─" * 54)
    known = latest_post_id(conn)
    rows = truth_feed.fetch_recent(pages=pages, stop_at_id=known, verbose=verbose)
    for r in rows:
        s = sig.extract(r["text"])
        r.update(s.as_row())
    added = insert_posts(conn, rows)
    if verbose:
        texted = sum(1 for r in rows if r["text"])
        print(f"  {len(rows)} fetched ({texted} with text), {added} new")
    return added


def make_predictions(conn, models, run_id, horizon, window_hours, verbose=True) -> int:
    if verbose:
        print("\n── predictions " + "─" * 55)
    since = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
    news_rows = recent_news(conn, since)
    headlines = predict.select_headlines(news_rows)

    if not headlines:
        if verbose:
            print("  no sufficiently relevant headlines in window — skipping")
        return 0

    if verbose:
        print(f"  conditioning on {len(headlines)} headlines:")
        for h in headlines[:5]:
            print(f"    • [{h['relevance']:.2f}] {h['title'][:70]}")

    added = 0
    for model in models:
        try:
            p = predict.make_prediction(model, headlines, horizon_hours=horizon,
                                        window_hours=window_hours)
            insert_prediction(conn, p.as_row(run_id))
            added += 1
            if verbose:
                print(f"  [{model}] {p.text[:88] or '(empty)'}")
                print(f"      topic={p.signal.topic} pol={p.signal.polarity:+.2f} "
                      f"tickers={list(p.signal.tickers)}")
        except Exception as exc:
            # A model failing must not lose the cycle's news or ground truth.
            print(f"  [{model}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    return added


def run_once(models, horizon, window_hours, pages=2, verbose=True) -> dict:
    run_id = f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    counts = {"news_added": 0, "posts_added": 0, "preds_added": 0, "matches_added": 0}

    if verbose:
        print("=" * 70)
        print(f"CYCLE {run_id}   {utcnow()}")
        print("=" * 70)

    with connect() as conn:
        start_run(conn, run_id)

    try:
        with connect() as conn:
            counts["news_added"], _ = collect_news(conn, verbose)
        with connect() as conn:
            counts["preds_added"] = make_predictions(
                conn, models, run_id, horizon, window_hours, verbose)
        # Posts before matching: score the just-elapsed horizon against the
        # freshest ground truth this cycle can see.
        with connect() as conn:
            counts["posts_added"] = collect_posts(conn, pages, verbose)

        if verbose:
            print("\n── matching " + "─" * 58)
        counts["matches_added"] = match.match_pending(verbose=verbose)

        with connect() as conn:
            finish_run(conn, run_id, status="ok", **counts)
    except Exception as exc:
        with connect() as conn:
            finish_run(conn, run_id, status="error", error=str(exc)[:500], **counts)
        raise

    if verbose:
        print("\n── totals " + "─" * 60)
        with connect() as conn:
            for k, v in stats(conn).items():
                print(f"  {k:>16}: {v}")
        print(f"\n  this cycle: {counts}")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="markov,ngram,rag_retrieve",
                    help="comma-separated; " + ",".join(sorted(predict.PREDICTORS)))
    ap.add_argument("--horizon", type=float, default=predict.DEFAULT_HORIZON_HOURS,
                    help="hours a prediction claims to cover")
    ap.add_argument("--window", type=float, default=predict.DEFAULT_NEWS_WINDOW_HOURS,
                    help="hours of news to condition on")
    ap.add_argument("--pages", type=int, default=2, help="Truth Social pages per cycle")
    ap.add_argument("--daemon", action="store_true", help="loop instead of exiting")
    ap.add_argument("--interval", type=int, default=1800, help="daemon seconds between cycles")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in predict.PREDICTORS]
    if unknown:
        ap.error(f"unknown models {unknown}; known: {sorted(predict.PREDICTORS)}")

    init_db()
    verbose = not args.quiet

    if not args.daemon:
        run_once(models, args.horizon, args.window, args.pages, verbose)
        return

    print(f"daemon: {len(models)} models, every {args.interval}s. Ctrl-C to stop.\n")
    while True:
        try:
            run_once(models, args.horizon, args.window, args.pages, verbose)
        except KeyboardInterrupt:
            print("\nstopped")
            return
        except Exception:
            # Keep the loop alive: a transient feed or API failure should not end
            # a collection run that may have been going for weeks.
            traceback.print_exc()
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped")
            return


if __name__ == "__main__":
    main()
