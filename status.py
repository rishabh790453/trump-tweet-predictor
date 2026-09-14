#!/usr/bin/env python3
"""
Pipeline status — the daily-driver command.

    venv_mac/bin/python3 status.py

Shows what has been collected, whether collection is actually running, how the
models are scoring, and whether there is yet enough data to say anything.
"""

import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

from pipeline import score as sc
from pipeline.store import connect, stats


def _age(iso: str) -> str:
    if not iso or iso == "-":
        return "never"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso[:16]
    h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    if h < 1:
        return f"{h * 60:.0f}m ago"
    if h < 48:
        return f"{h:.1f}h ago"
    return f"{h / 24:.1f}d ago"


def main() -> int:
    with connect() as conn:
        st = stats(conn)

        print("=" * 68)
        print("  TRUMP POST PREDICTION PIPELINE — STATUS")
        print("=" * 68)

        print("\nCOLLECTED")
        print(f"  news headlines      {st['news']:>7}")
        print(f"  posts (ground truth){st['posts']:>7}   "
              f"{st['posts_with_text']} with text")
        print(f"  predictions         {st['predictions']:>7}")
        print(f"  scored outcomes     {st['matches']:>7}")
        bars = conn.execute("SELECT COUNT(*) FROM market").fetchone()[0]
        print(f"  market bars         {bars:>7}")

        print(f"\n  newest post         {_age(st['newest_post'])}")
        print(f"  oldest post         {_age(st['oldest_post'])}")

        # Collection health
        last = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        if last:
            print(f"  last cycle          {_age(last['started_at'])}  "
                  f"status={last['status']}")
            if last["error"]:
                print(f"    error: {last['error'][:60]}")
        else:
            print("  last cycle          never — run run_cycle.py")

        runs_24h = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE started_at >= ?",
            ((datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),),
        ).fetchone()[0]
        print(f"  cycles last 24h     {runs_24h}")

        # Gaps in ground truth: the cost of the Mac being asleep
        posts = conn.execute(
            "SELECT created_at FROM posts ORDER BY created_at").fetchall()
        if len(posts) > 2:
            times = [datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
                     for p in posts]
            gaps = [(b - a).total_seconds() / 3600 for a, b in zip(times, times[1:])]
            biggest = max(gaps)
            print(f"  largest post gap    {biggest:.1f}h "
                  f"({'normal — he sleeps too' if biggest < 24 else 'possible collection gap'})")

        # Scoring
        rows = conn.execute(
            """SELECT p.model, m.* FROM matches m
               JOIN predictions p ON p.id = m.prediction_id""").fetchall()
        if rows:
            print("\nMODEL SCORES  (topic/direction accuracy are over matched rows only;")
            print("               read them together with post_rate)")
            by_model: dict[str, list[dict]] = {}
            for r in rows:
                by_model.setdefault(r["model"], []).append(dict(r))
            print(f"  {'model':<18}{'n':>5}{'post%':>8}{'topic%':>8}"
                  f"{'dir%':>7}{'rouge1':>9}")
            print("  " + "-" * 55)
            for model, rs in sorted(by_model.items()):
                s = sc.summarise(rs)
                fmt = lambda v: f"{v * 100:>7.1f}" if v is not None else "      -"
                print(f"  {model:<18}{s['n']:>5}{fmt(s['post_rate'])}"
                      f"{fmt(s['topic_accuracy'])}{fmt(s['direction_accuracy'])}"
                      f"{s['rouge1'] if s['rouge1'] is not None else '-':>9}")

            # Baseline: always guessing his most frequent topic
            topics = Counter(
                r["topic"] for r in conn.execute(
                    "SELECT topic FROM posts WHERE topic IS NOT NULL").fetchall()
            )
            if topics:
                base = sc.baseline_topic_accuracy(dict(topics))
                top = topics.most_common(1)[0]
                print(f"\n  baseline: always guess '{top[0]}' → {base * 100:.1f}% topic accuracy")
                print("  a model must clear this to have learned anything")
        else:
            print("\nMODEL SCORES        none yet — predictions score once their")
            print("                    horizon elapses (default 6h)")

        # Pending
        pending = conn.execute(
            """SELECT COUNT(*) FROM predictions p
               LEFT JOIN matches m ON m.prediction_id = p.id
               WHERE m.prediction_id IS NULL""").fetchone()[0]
        print(f"\n  predictions awaiting horizon: {pending}")

    # Readiness for inference
    from pipeline.backtest import MIN_EVENTS_FOR_INFERENCE, event_study
    res = event_study(verbose=False)
    n = res["overall"].get("n", 0)
    print(f"\nTRADEABILITY")
    print(f"  signal events with market data: {n}")
    print(f"  needed for inference:           ~{MIN_EVENTS_FOR_INFERENCE}")
    if n < MIN_EVENTS_FOR_INFERENCE:
        pct = n / MIN_EVENTS_FOR_INFERENCE * 100
        print(f"  progress: {pct:.0f}%  — keep collecting; results so far are noise")
    else:
        print("  enough events — run: venv_mac/bin/python3 -m pipeline.backtest --control")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
