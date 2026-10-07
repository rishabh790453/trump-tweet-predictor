"""
World news intensity from GDELT — a real multi-source measure.

Why this replaces news_data.db for volume work: that database is 97.5% New York
Times, and its article counts fall from 149k (2009) to 49k (2025), which is
archive-API coverage rather than anything about the world. Regressing post
counts on it was mostly regressing on our own scraper's history.

GDELT monitors thousands of outlets globally and its coverage is stable over
time — 1,432,550 article-mentions on 2026-09-02 against 1,433,650 on 2025-09-02.
That stability is the whole reason it can measure "is the world busy" when a
single-publisher feed cannot.

Access route matters. The DOC API (api.gdeltproject.org) rate-limits to one
request per 5s and durably 429s this network even on a single request after a
cooldown. The BULK path is unmetered and is what this module uses: GDELT 1.0
publishes one file per day at
    http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
~8MB zipped, ~120k event rows, 58 tab-separated columns.

Sampling: a full daily series 2022-2026 is ~11GB. News intensity is strongly
autocorrelated within a week, so for a WEEKLY regression one sampled weekday per
week is a legitimate (noisier) proxy at 1/7 the bandwidth. `--per-week` controls
this; pass 7 for the complete series when the bandwidth is worth it.
"""

import argparse
import io
import sqlite3
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

BASE = "http://data.gdeltproject.org/events"
HEADERS = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0.0.0 Safari/537.36")}
DB = Path(__file__).parent.parent / "gdelt_volume.db"

# GDELT 1.0 export schema, 0-indexed.
C_ACTOR1, C_ACTOR2 = 6, 16
C_EVENTCODE, C_QUADCLASS = 26, 29
C_NUMMENTIONS, C_NUMSOURCES, C_NUMARTICLES = 31, 32, 33
C_AVGTONE, C_SOURCEURL = 34, 57

SCHEMA = """
CREATE TABLE IF NOT EXISTS gdelt_daily (
    d              TEXT PRIMARY KEY,
    n_events       INTEGER,
    n_mentions     INTEGER,   -- total article-mentions: world news intensity
    n_articles     INTEGER,
    mean_tone      REAL,
    trump_events   INTEGER,   -- events naming Trump as an actor
    trump_mentions INTEGER,
    usa_events     INTEGER,   -- events with a US actor
    fetched_at     TEXT
);
"""


def _conn():
    c = sqlite3.connect(str(DB), timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(SCHEMA)
    return c


def have(d: date) -> bool:
    with _conn() as c:
        return c.execute("SELECT 1 FROM gdelt_daily WHERE d=?",
                         (d.isoformat(),)).fetchone() is not None


def fetch_day(d: date, timeout: int = 120) -> dict | None:
    """Download and aggregate one day. Returns None if GDELT has no such file."""
    url = f"{BASE}/{d:%Y%m%d}.export.CSV.zip"
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
    except Exception as exc:
        print(f"  {d} ERR {type(exc).__name__}", flush=True)
        return None
    if r.status_code != 200:
        print(f"  {d} HTTP {r.status_code}", flush=True)
        return None

    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        raw = z.read(z.namelist()[0]).decode("utf-8", "replace")
    except Exception as exc:
        print(f"  {d} unzip failed: {type(exc).__name__}", flush=True)
        return None

    n_ev = n_men = n_art = 0
    t_ev = t_men = usa = 0
    tone_sum = tone_n = 0.0

    for line in raw.split("\n"):
        if not line:
            continue
        f = line.split("\t")
        if len(f) <= C_SOURCEURL:
            continue
        n_ev += 1

        def num(i):
            try:
                return int(f[i])
            except (ValueError, IndexError):
                return 0

        m, a = num(C_NUMMENTIONS), num(C_NUMARTICLES)
        n_men += m
        n_art += a
        try:
            tone_sum += float(f[C_AVGTONE]); tone_n += 1
        except (ValueError, IndexError):
            pass

        a1, a2 = f[C_ACTOR1].upper(), f[C_ACTOR2].upper()
        if "TRUMP" in a1 or "TRUMP" in a2:
            t_ev += 1
            t_men += m
        # Country codes sit next to the actor names; a cheap USA flag.
        if "USA" in line[:400]:
            usa += 1

    return {"d": d.isoformat(), "n_events": n_ev, "n_mentions": n_men,
            "n_articles": n_art,
            "mean_tone": round(tone_sum / tone_n, 4) if tone_n else None,
            "trump_events": t_ev, "trump_mentions": t_men, "usa_events": usa,
            "fetched_at": datetime.utcnow().isoformat()}


def store(row: dict) -> None:
    with _conn() as c:
        c.execute("""INSERT OR REPLACE INTO gdelt_daily
            (d,n_events,n_mentions,n_articles,mean_tone,trump_events,
             trump_mentions,usa_events,fetched_at)
            VALUES (:d,:n_events,:n_mentions,:n_articles,:mean_tone,
                    :trump_events,:trump_mentions,:usa_events,:fetched_at)""", row)


def sample_days(start: date, end: date, per_week: int) -> list[date]:
    """Pick `per_week` weekdays from each week in the range (Wed-centred)."""
    # Wed first, then Sat/Mon, then fill — so a 1/week sample is mid-week.
    order = [2, 5, 0, 3, 6, 1, 4]
    keep = set(order[:max(1, min(per_week, 7))])
    out, d = [], start
    while d <= end:
        if d.weekday() in keep:
            out.append(d)
        d += timedelta(days=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--per-week", type=int, default=2,
                    help="weekdays sampled per week (7 = full series, ~8MB/day)")
    ap.add_argument("--pause", type=float, default=0.5)
    args = ap.parse_args()

    s = date.fromisoformat(args.start); e = date.fromisoformat(args.end)
    days = [d for d in sample_days(s, e, args.per_week) if not have(d)]
    print(f"GDELT daily volume: {len(days)} days to fetch "
          f"({args.per_week}/week, {s} .. {e}) ~{len(days)*8/1000:.1f} GB", flush=True)

    ok = 0
    for i, d in enumerate(days, 1):
        row = fetch_day(d)
        if row:
            store(row); ok += 1
            if ok % 10 == 0 or i == len(days):
                print(f"  [{i}/{len(days)}] {d} events={row['n_events']:,} "
                      f"mentions={row['n_mentions']:,} trump_ev={row['trump_events']:,}",
                      flush=True)
        time.sleep(args.pause)

    with _conn() as c:
        n, lo, hi = c.execute(
            "SELECT COUNT(*),MIN(d),MAX(d) FROM gdelt_daily").fetchone()
    print(f"\nstored {ok} new; db now holds {n} days ({lo} .. {hi}) -> {DB}")


if __name__ == "__main__":
    main()
