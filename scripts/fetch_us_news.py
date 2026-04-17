"""
US-Focused News Fetcher
========================
Sources:
  - GDELT v2  : 2015-2026, indexes Reuters/AP/Fox/CNN/WashPost/NYT/Bloomberg etc.
                Free, no key. Max 250 results per query. Rate limit: 1 req / 5s.
  - NYT Archive: 2009-2026, every NYT article. Free key from developer.nytimes.com.
                 ~2,000-4,000 articles/month. Rate limit: 10 req/min.

Run:
    py -3 fetch_us_news.py --gdelt          # GDELT 2015-2026 (no key needed)
    py -3 fetch_us_news.py --nyt            # NYT 2009-2026 (needs nytimes_key in config)
    py -3 fetch_us_news.py --all            # both
    py -3 fetch_us_news.py --stats          # show DB stats

Checkpoint file: us_news_progress.json  (resume-safe, saves after every task)

Expected yield:
  GDELT  : 100,000 - 250,000 articles (2015-2026)
  NYT    : 200,000 - 400,000 articles (2009-2026, ~2500/month)
"""

import requests
import json
import sqlite3
import hashlib
import time
import sys
import os
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# ── Config ───────────────────────────────────────────────────────────────────
DB_PATH    = "news_data.db"
CONFIG     = "news_api_config.json"
CHECKPOINT = "us_news_progress.json"

GDELT_START = date(2015, 2, 19)   # GDELT v2 full-text begins here
GDELT_END   = date(2026, 3, 1)
NYT_START   = date(2009, 1, 1)    # matches earliest Trump tweet
NYT_END     = date(2026, 3, 1)

# 10 topic queries focused on what Trump tweets about
# Each query runs per-month to maximise article yield
GDELT_QUERIES = [
    ("us_economy",    "(economy OR recession OR inflation OR GDP OR unemployment) sourcecountry:US"),
    ("stock_market",  "(stock market OR wall street OR dow jones OR S&P OR NASDAQ OR shares) sourcecountry:US"),
    ("fed",           "(Federal Reserve OR interest rates OR Jerome Powell OR Janet Yellen OR monetary policy) sourcecountry:US"),
    ("trade",         "(trade war OR tariffs OR trade deal OR trade deficit OR imports exports) sourcecountry:US"),
    ("us_politics",   "(congress OR senate OR White House OR Republican OR Democrat OR election OR midterm) sourcecountry:US"),
    ("trump",         "Trump sourcecountry:US"),
    ("energy_oil",    "(oil prices OR gas prices OR energy prices OR OPEC OR shale) sourcecountry:US"),
    ("geopolitics",   "(China OR Russia OR North Korea OR Iran OR NATO OR Ukraine OR Middle East) sourcecountry:US"),
    ("pandemic",      "(COVID OR pandemic OR coronavirus OR vaccine OR Fauci) sourcecountry:US"),
    ("finance",       "(banking OR financial crisis OR federal debt OR treasury OR bonds OR hedge fund) sourcecountry:US"),
]


# ── Database ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id TEXT UNIQUE,
            title TEXT,
            description TEXT,
            content TEXT,
            source TEXT,
            author TEXT,
            url TEXT UNIQUE,
            published_at TEXT,
            fetched_at TEXT DEFAULT (datetime('now')),
            category TEXT,
            sentiment_score REAL,
            keywords TEXT,
            image_url TEXT,
            lang TEXT DEFAULT 'en'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pub ON news_articles(published_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_src ON news_articles(source)")
    conn.commit()
    return conn


def insert_batch(conn, articles):
    inserted = 0
    cur = conn.cursor()
    for a in articles:
        try:
            cur.execute("""
                INSERT OR IGNORE INTO news_articles
                  (article_id, title, description, content, source, author, url,
                   published_at, category, keywords, image_url, lang)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                a.get("article_id") or _make_id(a.get("url",""), a.get("title","")),
                a.get("title","")[:500],
                a.get("description","")[:1000],
                a.get("content","")[:8000],
                a.get("source",""),
                a.get("author",""),
                a.get("url",""),
                a.get("published_at",""),
                a.get("category",""),
                json.dumps(a.get("keywords", [])),
                a.get("image_url",""),
                a.get("lang","en"),
            ))
            if cur.rowcount > 0:
                inserted += 1
        except Exception:
            continue
    conn.commit()
    return inserted


def db_total(conn):
    return conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]


def _make_id(url, title):
    return hashlib.md5(f"{url}{title}".encode()).hexdigest()


# ── Checkpoint ────────────────────────────────────────────────────────────────
def load_cp():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"gdelt_done": [], "nyt_done": [], "total_new": 0}


def save_cp(cp):
    with open(CHECKPOINT, "w") as f:
        json.dump(cp, f, indent=2)


# ── GDELT ─────────────────────────────────────────────────────────────────────
def _gdelt_fetch(query, start_s, end_s, max_records=250):
    """Single GDELT request. Returns list of articles or [] on error/rate-limit."""
    params = {
        "query":         query,
        "mode":          "artlist",
        "maxrecords":    min(max_records, 250),
        "format":        "json",
        "startdatetime": start_s,
        "enddatetime":   end_s,
    }
    for attempt in range(4):
        try:
            r = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params=params,
                timeout=30,
            )
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  [GDELT 429 – waiting {wait}s]", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
        except Exception:
            return []

        out = []
        for item in data.get("articles", []):
            pub = item.get("seendate", "")
            if len(pub) == 14:
                try:
                    pub = datetime.strptime(pub, "%Y%m%d%H%M%S").isoformat()
                except Exception:
                    pass
            out.append({
                "title":        item.get("title", ""),
                "source":       item.get("domain", "GDELT"),
                "url":          item.get("url", ""),
                "published_at": pub,
                "image_url":    item.get("socialimage", ""),
                "category":     "gdelt-us",
                "lang":         item.get("language", "en"),
            })
        return out

    return []


def run_gdelt(conn, cp):
    print("\n" + "="*70)
    print("GDELT — US sources, 10 topic queries × monthly, 2015-2026")
    print("  Sources: Reuters, AP, Fox News, CNN, WashPost, Bloomberg, etc.")
    print("  Rate limit: 6 seconds between requests")
    print("="*70)

    months = []
    cur = GDELT_START.replace(day=1)
    while cur < GDELT_END:
        months.append((cur.year, cur.month))
        cur += relativedelta(months=1)

    total_months = len(months)
    grand_new = 0

    for qi, (qname, qstr) in enumerate(GDELT_QUERIES):
        q_new = 0
        print(f"\n  [{qi+1}/{len(GDELT_QUERIES)}] {qname}")
        for mi, (yr, mo) in enumerate(months):
            key = f"gdelt|{qname}|{yr}-{mo:02d}"
            if key in cp["gdelt_done"]:
                continue

            # Build datetime strings
            start_dt = datetime(yr, mo, 1)
            if mo == 12:
                end_dt = datetime(yr+1, 1, 1)
            else:
                end_dt = datetime(yr, mo+1, 1)
            start_s = start_dt.strftime("%Y%m%d%H%M%S")
            end_s   = end_dt.strftime("%Y%m%d%H%M%S")

            arts = _gdelt_fetch(qstr, start_s, end_s)
            new  = insert_batch(conn, arts) if arts else 0

            q_new    += new
            grand_new += new
            cp["gdelt_done"].append(key)
            cp["total_new"] += new
            save_cp(cp)

            pct = (mi+1) / total_months * 100
            if new > 0 or (mi+1) % 24 == 0:
                print(f"    {yr}-{mo:02d}  {pct:5.1f}%  +{new:3d}  DB:{db_total(conn):,}", flush=True)

            time.sleep(6)  # GDELT: 1 request per 5 seconds, use 6 to be safe

        print(f"    done — {q_new:,} new articles for [{qname}]")

    print(f"\n  GDELT complete — {grand_new:,} total new articles")
    return grand_new


# ── NYT Archive API ───────────────────────────────────────────────────────────
def _nyt_fetch_month(year, month, api_key):
    """
    Fetch ALL NYT articles for a given month via Archive API.
    Returns typically 2,000-4,000 articles. Rate limit: 10 req/min → 6s sleep.
    """
    url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
    for attempt in range(3):
        try:
            r = requests.get(url, params={"api-key": api_key}, timeout=60)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  [NYT 429 – waiting {wait}s]", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            docs = r.json().get("response", {}).get("docs", [])
        except Exception as e:
            print(f"  [NYT error {yr}/{mo}: {e}]")
            return []

        out = []
        for doc in docs:
            # Filter: only keep US-relevant sections
            section = doc.get("section_name", "")
            desk    = doc.get("news_desk", "")
            kws     = [kw.get("value","") for kw in (doc.get("keywords") or [])]
            out.append({
                "article_id":   doc.get("_id", ""),
                "title":        doc.get("headline", {}).get("main", ""),
                "description":  doc.get("abstract", "") or doc.get("snippet", ""),
                "content":      doc.get("lead_paragraph", ""),
                "source":       "New York Times",
                "author":       doc.get("byline", {}).get("original", ""),
                "url":          doc.get("web_url", ""),
                "published_at": doc.get("pub_date", ""),
                "category":     section or desk or "general",
                "keywords":     kws or [],
                "lang":         "en",
            })
        return out

    return []


def run_nyt(conn, cp):
    print("\n" + "="*70)
    print("NYT Archive — every NYT article, 2009-2026")
    print("  ~2,000-4,000 articles per month → ~300,000-600,000 total")
    print("  Rate limit: 6 seconds between requests (10 req/min max)")
    print("="*70)

    config = {}
    if os.path.exists(CONFIG):
        with open(CONFIG) as f:
            config = json.load(f)

    api_key = config.get("nytimes_key", "").strip()
    if not api_key:
        print("""
  *** NYT API KEY MISSING ***

  Get a free key in ~2 minutes:
    1. Go to: https://developer.nytimes.com/
    2. Sign in / create account
    3. Go to "My Apps" → create a new app
    4. Enable "Archive API"
    5. Copy the API key
    6. Paste it in news_api_config.json:
         "nytimes_key": "your_key_here"
    7. Re-run: py -3 fetch_us_news.py --nyt
""")
        return 0

    months = []
    cur = NYT_START.replace(day=1)
    while cur < NYT_END:
        months.append((cur.year, cur.month))
        cur += relativedelta(months=1)

    total_months = len(months)
    grand_new = 0

    for mi, (yr, mo) in enumerate(months):
        key = f"nyt|{yr}-{mo:02d}"
        if key in cp["nyt_done"]:
            continue

        arts = _nyt_fetch_month(yr, mo, api_key)
        new  = insert_batch(conn, arts) if arts else 0

        grand_new += new
        cp["nyt_done"].append(key)
        cp["total_new"] += new
        save_cp(cp)

        pct = (mi+1) / total_months * 100
        print(f"  {yr}-{mo:02d}  {pct:5.1f}%  +{new:4d} articles  DB:{db_total(conn):,}", flush=True)

        time.sleep(6)  # NYT: 10 req/min

    print(f"\n  NYT complete — {grand_new:,} total new articles")
    return grand_new


# ── Stats ─────────────────────────────────────────────────────────────────────
def print_stats(conn):
    total = db_total(conn)
    print(f"\n{'='*70}")
    print(f"DATABASE STATS  ({total:,} total articles)")
    print(f"{'='*70}")

    rows = conn.execute("""
        SELECT source, COUNT(*) n FROM news_articles
        GROUP BY source ORDER BY n DESC LIMIT 15
    """).fetchall()
    print("\n  Top sources:")
    for src, n in rows:
        print(f"    {n:>8,}  {src}")

    rows = conn.execute("""
        SELECT substr(published_at,1,4) yr, COUNT(*) n
        FROM news_articles WHERE published_at GLOB '20*'
        GROUP BY yr ORDER BY yr
    """).fetchall()
    print("\n  Articles per year:")
    for yr, n in rows:
        bar = "#" * min(n // 2000, 30)
        print(f"    {yr}  {n:>7,}  {bar}")

    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--all"
    conn = get_db()
    cp   = load_cp()

    if mode == "--stats":
        print_stats(conn)
        conn.close()
        return

    gdelt_done = len(cp["gdelt_done"])
    nyt_done   = len(cp["nyt_done"])
    print(f"\nDB: {db_total(conn):,} articles  |  Checkpoint: {gdelt_done} GDELT + {nyt_done} NYT tasks done")

    t0 = time.time()

    if mode in ("--all", "--gdelt"):
        run_gdelt(conn, cp)

    if mode in ("--all", "--nyt"):
        run_nyt(conn, cp)

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} minutes")
    print_stats(conn)
    conn.close()


if __name__ == "__main__":
    main()
