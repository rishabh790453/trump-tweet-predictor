"""
Comprehensive News Data Fetcher
================================
Targets: US news, global events, US economy & stock market
Sources: GDELT (free, no key), Guardian API (key in config)
Coverage: 2009-2026 (matches Trump tweet data)
Expected yield: 200,000 - 500,000 articles

Features:
- Checkpoint/resume: saves progress so you can stop and restart
- Multi-query GDELT: 10 topic clusters per month
- Guardian by section + full pagination (not capped at 20 pages)
- All articles deduplicated and stored in news_data.db

Run:
    py -3 fetch_comprehensive_news.py                  # all phases
    py -3 fetch_comprehensive_news.py --gdelt-only     # GDELT only
    py -3 fetch_comprehensive_news.py --guardian-only  # both Guardian phases
    py -3 fetch_comprehensive_news.py --guardian-sections  # section-based only
    py -3 fetch_comprehensive_news.py --guardian-keywords  # keyword-based only
    py -3 fetch_comprehensive_news.py --stats          # show DB stats
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

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH        = "news_data.db"
CONFIG_FILE    = "news_api_config.json"
CHECKPOINT     = "fetch_progress.json"

START_DATE     = date(2009, 1, 1)   # Trump's first tweet was May 2009
END_DATE       = date(2026, 3, 1)   # Through March 2026

# GDELT queries: 10 topic clusters that cover what Trump tweets about
GDELT_QUERIES = [
    ("economy",      "economy OR recession OR inflation OR GDP OR unemployment OR jobs"),
    ("markets",      "stock market OR wall street OR dow jones OR S&P 500 OR NASDAQ OR shares"),
    ("fed",          "Federal Reserve OR interest rates OR monetary policy OR Jerome Powell OR Janet Yellen"),
    ("trade",        "trade war OR tariffs OR trade deal OR WTO OR imports OR exports OR deficit"),
    ("trump_pols",   "Trump OR White House OR congress OR senate OR Republican OR Democrat OR election"),
    ("energy",       "oil prices OR energy prices OR gas prices OR OPEC OR crude oil"),
    ("geopolitics",  "China OR Russia OR North Korea OR Iran OR NATO OR Ukraine OR Middle East"),
    ("pandemic",     "COVID OR pandemic OR coronavirus OR vaccine OR Fauci OR lockdown"),
    ("finance",      "banking OR financial crisis OR federal debt OR treasury bonds OR hedge fund"),
    ("media_media",  "fake news OR media OR twitter OR social media OR Fox News OR CNN MSNBC"),
]

# Guardian sections most relevant to the research
GUARDIAN_SECTIONS = [
    "us-news",
    "business",
    "world",
    "politics",
    "money",
    "technology",
    "environment",
]

# Guardian keyword queries for a second targeted pass (US economy / markets focus)
GUARDIAN_KEYWORD_QUERIES = [
    ("us_economy",    "US economy OR American economy OR GDP growth OR US recession"),
    ("stock_market",  "stock market OR wall street OR dow jones OR S&P 500 OR NASDAQ"),
    ("fed_reserve",   "Federal Reserve OR interest rates OR Jerome Powell OR Janet Yellen OR quantitative easing"),
    ("trade_tariffs", "trade war OR US tariffs OR China trade OR trade deficit OR WTO"),
    ("trump_policy",  "Trump policy OR Trump administration OR Trump tweet OR Trump executive order"),
    ("inflation",     "US inflation OR consumer prices OR CPI OR cost of living"),
    ("jobs_economy",  "US jobs OR unemployment rate OR jobs report OR payroll OR labor market"),
    ("energy_oil",    "oil prices OR gas prices OR energy prices OR OPEC OR shale"),
    ("banking",       "US banking OR financial crisis OR federal debt OR US bonds OR Wall Street bailout"),
    ("geopolitics",   "US foreign policy OR North Korea OR Iran sanctions OR Russia Ukraine OR China relations"),
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
    """Insert articles, skip duplicates. Returns count of new articles inserted."""
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
                a.get("article_id") or make_id(a.get("url",""), a.get("title","")),
                a.get("title",""),
                a.get("description","")[:1000],
                a.get("content","")[:5000],
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


def make_id(url, title):
    return hashlib.md5(f"{url}{title}".encode()).hexdigest()


# ── Checkpoint ───────────────────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"gdelt_done": [], "guardian_done": [], "total_inserted": 0}


def save_checkpoint(cp):
    with open(CHECKPOINT, "w") as f:
        json.dump(cp, f, indent=2)


# ── GDELT ─────────────────────────────────────────────────────────────────────
def fetch_gdelt_month(query_str, year, month, max_records=250, retries=3):
    """
    Fetch up to 250 GDELT articles for a topic within a calendar month.
    Handles 429 rate-limit responses with exponential backoff.
    """
    start_dt = datetime(year, month, 1)
    if month == 12:
        end_dt = datetime(year + 1, 1, 1)
    else:
        end_dt = datetime(year, month + 1, 1)

    start_s = start_dt.strftime("%Y%m%d%H%M%S")
    end_s   = end_dt.strftime("%Y%m%d%H%M%S")

    params = {
        "query":         query_str,
        "mode":          "artlist",
        "maxrecords":    min(max_records, 250),
        "format":        "json",
        "startdatetime": start_s,
        "enddatetime":   end_s,
    }

    for attempt in range(retries):
        try:
            r = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params=params,
                timeout=30,
            )
            if r.status_code == 429:
                wait = 30 * (attempt + 1)  # 30s, 60s, 90s
                print(f" [GDELT 429 - waiting {wait}s]", end="", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
        except Exception:
            return []

        articles = []
        for item in data.get("articles", []):
            pub = item.get("seendate", "")
            if len(pub) == 14:
                try:
                    pub = datetime.strptime(pub, "%Y%m%d%H%M%S").isoformat()
                except Exception:
                    pass
            articles.append({
                "title":        item.get("title", ""),
                "description":  "",
                "content":      "",
                "source":       item.get("domain", "GDELT"),
                "author":       "",
                "url":          item.get("url", ""),
                "published_at": pub,
                "image_url":    item.get("socialimage", ""),
                "category":     "gdelt",
                "lang":         item.get("language", "en"),
            })
        return articles

    return []  # all retries exhausted


def run_gdelt(conn, cp, start=START_DATE, end=END_DATE):
    print("\n" + "="*70)
    print("PHASE 1: GDELT — 10 topic queries × every month")
    print("="*70)

    months = []
    cur = start.replace(day=1)
    while cur < end:
        months.append((cur.year, cur.month))
        cur += relativedelta(months=1)

    total_months = len(months)
    grand_new = 0

    for qi, (qname, qstr) in enumerate(GDELT_QUERIES):
        print(f"\n  Query {qi+1}/{len(GDELT_QUERIES)}: [{qname}]")
        for mi, (yr, mo) in enumerate(months):
            key = f"gdelt|{qname}|{yr}-{mo:02d}"
            if key in cp["gdelt_done"]:
                continue

            arts = fetch_gdelt_month(qstr, yr, mo)
            new  = insert_batch(conn, arts) if arts else 0
            grand_new += new
            cp["gdelt_done"].append(key)
            cp["total_inserted"] += new

            save_checkpoint(cp)
            pct = (mi + 1) / total_months * 100
            if (mi + 1) % 12 == 0 or mi == total_months - 1 or new > 0:
                print(f"    {yr}-{mo:02d}  {pct:5.1f}%  +{new:3d} new  DB total: {db_total(conn):,}")

            time.sleep(6)  # GDELT rate limit: max 1 req / 5 sec, use 6 to be safe

    print(f"\n  GDELT done — {grand_new:,} new articles added")
    return grand_new


# ── Guardian ──────────────────────────────────────────────────────────────────
def fetch_guardian_section_month(section, year, month, api_key, page_limit=50):
    """
    Fetch ALL Guardian articles in a section for a calendar month.
    page_limit=50 → up to 2,500 articles per section/month (plenty).
    """
    if month == 12:
        end_day = 31
    else:
        import calendar
        end_day = calendar.monthrange(year, month)[1]

    from_date = f"{year}-{month:02d}-01"
    to_date   = f"{year}-{month:02d}-{end_day:02d}"

    base_params = {
        "section":     section,
        "from-date":   from_date,
        "to-date":     to_date,
        "page-size":   50,
        "show-fields": "headline,bodyText,byline,thumbnail",
        "api-key":     api_key,
        "order-by":    "oldest",
    }

    articles = []
    total_pages = None

    for page in range(1, page_limit + 1):
        params = {**base_params, "page": page}
        try:
            r = requests.get(
                "https://content.guardianapis.com/search",
                params=params,
                timeout=15,
            )
            r.raise_for_status()
            data = r.json().get("response", {})
        except Exception as e:
            break

        if total_pages is None:
            total_pages = min(data.get("pages", 1), page_limit)

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            fields = item.get("fields", {})
            body   = fields.get("bodyText", "")
            articles.append({
                "article_id":   item.get("id", ""),
                "title":        fields.get("headline", item.get("webTitle", "")),
                "description":  body[:500],
                "content":      body[:5000],
                "source":       "The Guardian",
                "author":       fields.get("byline", ""),
                "url":          item.get("webUrl", ""),
                "published_at": item.get("webPublicationDate", ""),
                "image_url":    fields.get("thumbnail", ""),
                "category":     item.get("sectionName", section),
                "lang":         "en",
            })

        if page >= total_pages:
            break

        time.sleep(0.5)  # Guardian allows ~12 req/s on open platform

    return articles


def fetch_guardian_keyword_year(query_str, year, api_key, page_limit=100):
    """
    Fetch Guardian articles matching a keyword query for a full year.
    Uses year-long window; up to page_limit pages (50 articles each).
    """
    from_date = f"{year}-01-01"
    to_date   = f"{year}-12-31"

    base_params = {
        "q":           query_str,
        "from-date":   from_date,
        "to-date":     to_date,
        "page-size":   50,
        "show-fields": "headline,bodyText,byline,thumbnail",
        "api-key":     api_key,
        "order-by":    "oldest",
    }

    articles = []
    total_pages = None

    for page in range(1, page_limit + 1):
        params = {**base_params, "page": page}
        try:
            r = requests.get(
                "https://content.guardianapis.com/search",
                params=params,
                timeout=15,
            )
            r.raise_for_status()
            data = r.json().get("response", {})
        except Exception:
            break

        if total_pages is None:
            total_pages = min(data.get("pages", 1), page_limit)

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            fields = item.get("fields", {})
            body   = fields.get("bodyText", "")
            articles.append({
                "article_id":   item.get("id", ""),
                "title":        fields.get("headline", item.get("webTitle", "")),
                "description":  body[:500],
                "content":      body[:5000],
                "source":       "The Guardian",
                "author":       fields.get("byline", ""),
                "url":          item.get("webUrl", ""),
                "published_at": item.get("webPublicationDate", ""),
                "image_url":    fields.get("thumbnail", ""),
                "category":     item.get("sectionName", "general"),
                "lang":         "en",
            })

        if page >= total_pages:
            break

        time.sleep(0.5)

    return articles


def run_guardian_keywords(conn, cp, start=START_DATE, end=END_DATE):
    """Second Guardian pass: keyword-based queries per year for US-focused topics."""
    print("\n" + "="*70)
    print("PHASE 3: Guardian — US-focused keyword queries per year")
    print("="*70)

    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            config = json.load(f)

    api_key = config.get("guardian_key", "")
    if not api_key:
        print("  No Guardian API key — skipping.")
        return 0

    years = list(range(start.year, end.year + 1))
    grand_new = 0

    for qi, (qname, qstr) in enumerate(GUARDIAN_KEYWORD_QUERIES):
        print(f"\n  Query {qi+1}/{len(GUARDIAN_KEYWORD_QUERIES)}: [{qname}]")
        for yr in years:
            key = f"guardian_kw|{qname}|{yr}"
            if key in cp.get("guardian_kw_done", []):
                continue

            arts = fetch_guardian_keyword_year(qstr, yr, api_key)
            new  = insert_batch(conn, arts) if arts else 0
            grand_new += new

            cp.setdefault("guardian_kw_done", []).append(key)
            cp["total_inserted"] += new
            save_checkpoint(cp)
            if new > 0:
                print(f"    {yr}  +{new:4d} new  DB total: {db_total(conn):,}")

            time.sleep(0.3)

    print(f"\n  Guardian keyword pass done — {grand_new:,} new articles added")
    return grand_new


def run_guardian(conn, cp, start=START_DATE, end=END_DATE):
    print("\n" + "="*70)
    print("PHASE 2: Guardian — 7 sections × every month, full pagination")
    print("="*70)

    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            config = json.load(f)

    api_key = config.get("guardian_key", "")
    if not api_key:
        print("  No Guardian API key found in news_api_config.json — skipping.")
        return 0

    months = []
    cur = start.replace(day=1)
    while cur < end:
        months.append((cur.year, cur.month))
        cur += relativedelta(months=1)

    grand_new = 0

    for si, section in enumerate(GUARDIAN_SECTIONS):
        print(f"\n  Section {si+1}/{len(GUARDIAN_SECTIONS)}: [{section}]")
        section_new = 0

        for mi, (yr, mo) in enumerate(months):
            key = f"guardian|{section}|{yr}-{mo:02d}"
            if key in cp["guardian_done"]:
                continue

            arts = fetch_guardian_section_month(section, yr, mo, api_key)
            new  = insert_batch(conn, arts) if arts else 0
            section_new += new
            grand_new   += new
            cp["guardian_done"].append(key)
            cp["total_inserted"] += new

            save_checkpoint(cp)
            pct = (mi + 1) / len(months) * 100
            if (mi + 1) % 12 == 0 or mi == len(months) - 1 or new > 0:
                print(f"    {yr}-{mo:02d}  {pct:5.1f}%  +{new:3d} new  DB total: {db_total(conn):,}")

            time.sleep(0.2)  # small delay between months

        print(f"    [{section}] done — {section_new:,} new articles")

    print(f"\n  Guardian done — {grand_new:,} new articles added")
    return grand_new


# ── Stats ─────────────────────────────────────────────────────────────────────
def print_stats(conn):
    total = db_total(conn)
    print(f"\n{'='*70}")
    print(f"DATABASE STATS")
    print(f"{'='*70}")
    print(f"  Total articles: {total:,}")

    rows = conn.execute("""
        SELECT source, COUNT(*) as n
        FROM news_articles
        GROUP BY source
        ORDER BY n DESC
        LIMIT 15
    """).fetchall()
    print(f"\n  Top sources:")
    for src, n in rows:
        print(f"    {n:>8,}  {src}")

    rows = conn.execute("""
        SELECT category, COUNT(*) as n
        FROM news_articles
        GROUP BY category
        ORDER BY n DESC
        LIMIT 15
    """).fetchall()
    print(f"\n  Top categories:")
    for cat, n in rows:
        print(f"    {n:>8,}  {cat}")

    rows = conn.execute("""
        SELECT substr(published_at, 1, 4) as yr, COUNT(*) as n
        FROM news_articles
        WHERE published_at GLOB '20*'
        GROUP BY yr
        ORDER BY yr
    """).fetchall()
    print(f"\n  Articles per year:")
    for yr, n in rows:
        bar = "#" * (n // 500)
        print(f"    {yr}  {n:>7,}  {bar}")

    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--all"

    conn = get_db()
    cp   = load_checkpoint()

    if mode == "--stats":
        print_stats(conn)
        conn.close()
        return

    print(f"\nStarting fetch — current DB: {db_total(conn):,} articles")
    print(f"Checkpoint: {len(cp['gdelt_done'])} GDELT + {len(cp['guardian_done'])} Guardian tasks done")
    print(f"Target date range: {START_DATE} → {END_DATE}")

    t0 = time.time()

    if mode in ("--all", "--gdelt-only"):
        run_gdelt(conn, cp)

    if mode in ("--all", "--guardian-only", "--guardian-sections"):
        run_guardian(conn, cp)

    if mode in ("--all", "--guardian-only", "--guardian-keywords"):
        run_guardian_keywords(conn, cp)

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} minutes")
    print_stats(conn)
    conn.close()


if __name__ == "__main__":
    main()
