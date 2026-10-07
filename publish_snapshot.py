#!/usr/bin/env python3
"""
Publish a public snapshot of collected posts for the Vercel dashboard.

Why this exists: Truth Social sits behind Cloudflare and returns HTTP 403 ("Just
a moment...") to datacenter IPs, so the Vercel function cannot fetch posts
itself. It can reach news RSS and Kalshi fine — only Truth Social is blocked.

Residential IPs are not blocked, so the collector running on this machine keeps
working. This publishes what it has collected to a public GitHub gist, which
Vercel then reads. That also keeps the single-collector rule intact: the Mac
remains the only thing polling Truth Social.

    venv_mac/bin/python3 publish_snapshot.py --create   # first time
    venv_mac/bin/python3 publish_snapshot.py            # thereafter

Run it on a timer alongside the watcher; a few minutes of staleness is fine and
the page shows the snapshot age so a stalled feed is visible rather than silent.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pipeline import kalshi
from pipeline.store import connect, stats

REPO = Path(__file__).parent
GIST_ID_FILE = REPO / ".snapshot_gist_id"
SNAPSHOT_NAME = "trump_pipeline_snapshot.json"
MAX_POSTS = 40


def build_snapshot() -> dict:
    with connect() as conn:
        st = stats(conn)
        posts = [dict(r) for r in conn.execute(
            """SELECT id, created_at, text, has_media, is_reblog, favourites,
                      url, topic, polarity, intensity
               FROM posts ORDER BY created_at DESC LIMIT ?""", (MAX_POSTS,))]
        # run_id and headlines are what let the page group predictions under the
        # news they were conditioned on — every model in a run saw the same
        # headline set, which is the comparison the whole project is about.
        preds = [dict(r) for r in conn.execute(
            """SELECT p.run_id, p.model, p.predicted_at, p.text, p.topic,
                      p.polarity, p.headlines, p.horizon_hours,
                      m.matched, m.similarity, m.topic_hit
               FROM predictions p LEFT JOIN matches m ON m.prediction_id = p.id
               ORDER BY p.predicted_at DESC LIMIT 24""")]
        for pr in preds:
            try:
                pr["headlines"] = json.loads(pr["headlines"] or "[]")
            except (TypeError, ValueError):
                pr["headlines"] = []

    counts = kalshi.weekly_counts_from_db()
    this_week = kalshi.week_start_et().date().isoformat()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "local collector (residential IP)",
        "posts": posts,
        "predictions": preds,
        "week": {
            "week_start": this_week,
            "observed": counts.get(this_week, 0),
            "elapsed_pct": round(kalshi.elapsed_fraction() * 100, 1),
            # Explicitly flagged: the count only includes posts we were awake to
            # collect, so it is a floor, not the settlement figure.
            "is_partial": True,
        },
        "totals": {"posts": st["posts"], "news": st["news"],
                   "predictions": st["predictions"], "matches": st["matches"]},
    }


def _gh(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["gh"] + args, capture_output=True, text=True, **kw)


def publish(create: bool = False) -> str | None:
    snap = build_snapshot()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / SNAPSHOT_NAME
        path.write_text(json.dumps(snap, indent=1))

        if create or not GIST_ID_FILE.exists():
            r = _gh(["gist", "create", str(path), "--public",
                     "-d", "Trump post pipeline — live snapshot"])
            if r.returncode != 0:
                print(f"gist create failed: {r.stderr.strip()}", file=sys.stderr)
                return None
            url = r.stdout.strip().splitlines()[-1]
            gist_id = url.rstrip("/").split("/")[-1]
            GIST_ID_FILE.write_text(gist_id)
            print(f"created gist {gist_id}")
        else:
            gist_id = GIST_ID_FILE.read_text().strip()
            r = _gh(["gist", "edit", gist_id, "-f", SNAPSHOT_NAME, str(path)])
            if r.returncode != 0:
                print(f"gist edit failed: {r.stderr.strip()}", file=sys.stderr)
                return None

    raw = (f"https://gist.githubusercontent.com/rishabh790453/{gist_id}"
           f"/raw/{SNAPSHOT_NAME}")
    print(f"published {len(snap['posts'])} posts, week={snap['week']['observed']}")
    print(f"raw URL: {raw}")
    return raw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true", help="create the gist first time")
    args = ap.parse_args()
    sys.exit(0 if publish(args.create) else 1)
