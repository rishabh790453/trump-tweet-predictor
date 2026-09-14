#!/usr/bin/env python3
"""
Backfill historical Truth Social posts.

Worth running once before any analysis: the event study needs post history that
overlaps the market bars, and Yahoo only serves 5-minute bars for ~60 days. Going
back further than that buys posts that can only be studied at daily resolution.
"""
import argparse, sys
from pipeline import signal as sig, truth_feed
from pipeline.store import connect, init_db, insert_posts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=30, help="20 posts per page")
    args = ap.parse_args()

    init_db()
    print(f"Fetching up to {args.pages} pages ({args.pages * 20} posts)...\n")
    rows = truth_feed.fetch_recent(pages=args.pages, verbose=True)
    if not rows:
        print("nothing fetched"); return 1
    for r in rows:
        r.update(sig.extract(r["text"]).as_row())
    with connect() as conn:
        added = insert_posts(conn, rows)
    texted = sum(1 for r in rows if r["text"])
    print(f"\n{len(rows)} fetched, {texted} with text, {added} new")
    print(f"range: {min(r['created_at'] for r in rows)[:16]} .. "
          f"{max(r['created_at'] for r in rows)[:16]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
