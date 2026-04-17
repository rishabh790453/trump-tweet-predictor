"""
build_dataset.py
================
Creates the master aligned dataset for all models.

For each Trump tweet:
  - Pulls NYT articles published in the 24h window before the tweet
  - Computes aggregated news features (sentiment, topic signals, named entity counts)
  - Computes tweet features (sentiment, style, topic signals)
  - Merges in market data where available

Outputs:
  master_dataset.csv   -- all tweets with news + market features
  market_dataset.csv   -- subset with actual market data (4,956 rows)

Run:
  py -3 build_dataset.py
"""

import sqlite3
import csv
import json
import re
import math
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Config ───────────────────────────────────────────────────────────────────
TWEETS_CSV  = "trump_tweets_cleaned.csv"
FINANCE_CSV = "trump_tweets_finance_with_market_data.csv"
NEWS_DB     = "news_data.db"
OUT_MASTER  = "master_dataset.csv"
OUT_MARKET  = "market_dataset.csv"

NEWS_WINDOW_HOURS = 24   # look-back window: news published in this many hours before tweet

# Keyword signal dictionaries
ECONOMY_WORDS  = {"economy","economic","gdp","recession","inflation","unemployment",
                  "jobs","growth","deficit","debt","budget","fiscal","tax","tariff",
                  "trade","market","stock","wall street","dow","nasdaq","federal reserve",
                  "interest rate","fed","treasury","bonds","yield","oil","gas","energy"}

POLITICS_WORDS = {"congress","senate","democrat","republican","election","vote","president",
                  "white house","policy","legislation","bill","impeach","investigation","fbi",
                  "justice","court","supreme","constitution","amendment","executive","order"}

MEDIA_WORDS    = {"fake news","media","press","cnn","nytimes","washington post","fox",
                  "msnbc","nbc","abc","cbs","reporter","journalist","newspaper","television","tv"}

CHINA_WORDS    = {"china","chinese","beijing","xi jinping","hong kong","taiwan","trade war"}
RUSSIA_WORDS   = {"russia","russian","putin","moscow","ukraine","nato","sanctions","collusion"}

# ── Helpers ──────────────────────────────────────────────────────────────────
vader = SentimentIntensityAnalyzer()

def sentiment(text):
    if not text or not text.strip():
        return 0.0
    return vader.polarity_scores(text)["compound"]

def keyword_density(text, words):
    """Fraction of words in text matching keyword set (case-insensitive)."""
    if not text:
        return 0.0
    tl = text.lower()
    return sum(1 for w in words if w in tl) / max(len(words), 1)

def caps_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)

def parse_ts(ts_str):
    """Parse timestamp string to timezone-aware datetime."""
    if not ts_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str[:25], fmt[:len(ts_str[:25])])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    return None

# ── Step 1: Load and index news by date ─────────────────────────────────────
def load_news_index():
    """
    Load NYT articles and build a date-keyed index.
    Returns dict: 'YYYY-MM-DD' -> list of {title, sentiment, signals}
    """
    print("Loading NYT news articles...")
    conn = sqlite3.connect(NEWS_DB)

    # Only load relevant categories, skip Sports/Arts/Food/etc.
    SKIP_CATS = {
        "Sports","Arts","Fashion & Style","Food","Travel","Real Estate",
        "Movies","Theater","Music","Books","N.Y. / Region","New York",
        "Crosswords & Games","T Magazine","Style","Magazine"
    }

    cur = conn.execute("""
        SELECT title, published_at, category
        FROM news_articles
        WHERE source = 'New York Times'
          AND published_at IS NOT NULL
          AND published_at != ''
        ORDER BY published_at
    """)

    index = defaultdict(list)
    skipped = 0
    total = 0

    for title, pub, cat in cur:
        total += 1
        if cat in SKIP_CATS:
            skipped += 1
            continue
        if not title or not pub:
            continue

        # Parse date
        date_str = pub[:10]   # 'YYYY-MM-DD'
        if not date_str.startswith("20"):
            continue

        s = sentiment(title)
        index[date_str].append({
            "title":    title,
            "sent":     s,
            "econ":     keyword_density(title, ECONOMY_WORDS),
            "politics": keyword_density(title, POLITICS_WORDS),
            "media":    keyword_density(title, MEDIA_WORDS),
            "china":    1 if any(w in title.lower() for w in CHINA_WORDS) else 0,
            "russia":   1 if any(w in title.lower() for w in RUSSIA_WORDS) else 0,
            "trump":    1 if "trump" in title.lower() else 0,
        })

    conn.close()
    print(f"  Loaded {total - skipped:,} articles across {len(index):,} days (skipped {skipped:,} irrelevant categories)")
    return index


def get_news_features(tweet_dt, news_index, window_hours=24):
    """
    Aggregate news features from articles published in [tweet_dt - window, tweet_dt].
    """
    articles = []
    for h in range(window_hours + 1):
        day = (tweet_dt - timedelta(hours=h)).strftime("%Y-%m-%d")
        articles.extend(news_index.get(day, []))

    if not articles:
        return {
            "news_count": 0,
            "news_sent_mean": 0.0,
            "news_sent_std": 0.0,
            "news_sent_neg_pct": 0.0,
            "news_econ_intensity": 0.0,
            "news_politics_intensity": 0.0,
            "news_media_intensity": 0.0,
            "news_china_pct": 0.0,
            "news_russia_pct": 0.0,
            "news_trump_pct": 0.0,
        }

    sents   = [a["sent"]    for a in articles]
    n       = len(articles)
    return {
        "news_count":              n,
        "news_sent_mean":          round(np.mean(sents), 4),
        "news_sent_std":           round(np.std(sents), 4),
        "news_sent_neg_pct":       round(sum(1 for s in sents if s < -0.05) / n, 4),
        "news_econ_intensity":     round(np.mean([a["econ"]    for a in articles]), 4),
        "news_politics_intensity": round(np.mean([a["politics"] for a in articles]), 4),
        "news_media_intensity":    round(np.mean([a["media"]   for a in articles]), 4),
        "news_china_pct":          round(sum(a["china"]  for a in articles) / n, 4),
        "news_russia_pct":         round(sum(a["russia"] for a in articles) / n, 4),
        "news_trump_pct":          round(sum(a["trump"]  for a in articles) / n, 4),
    }


# ── Step 2: Compute tweet features ──────────────────────────────────────────
def get_tweet_features(row):
    text = row.get("content", "") or ""
    ts   = parse_ts(row.get("timestamp", ""))
    s    = sentiment(text)

    return {
        "tweet_date":         ts.strftime("%Y-%m-%d") if ts else "",
        "tweet_year":         ts.year if ts else None,
        "tweet_month":        ts.month if ts else None,
        "tweet_hour":         ts.hour if ts else None,
        "tweet_platform":     1 if row.get("platform","") == "Twitter" else 0,
        "tweet_length":       len(text),
        "tweet_word_count":   len(text.split()),
        "tweet_caps_ratio":   round(caps_ratio(text), 4),
        "tweet_exclamations": text.count("!"),
        "tweet_questions":    text.count("?"),
        "tweet_sent":         round(s, 4),
        "tweet_sent_pos":     1 if s > 0.05 else 0,
        "tweet_sent_neg":     1 if s < -0.05 else 0,
        "tweet_econ":         round(keyword_density(text, ECONOMY_WORDS), 4),
        "tweet_politics":     round(keyword_density(text, POLITICS_WORDS), 4),
        "tweet_media":        round(keyword_density(text, MEDIA_WORDS), 4),
        "tweet_china":        1 if any(w in text.lower() for w in CHINA_WORDS) else 0,
        "tweet_russia":       1 if any(w in text.lower() for w in RUSSIA_WORDS) else 0,
        "tweet_retweets":     int(row.get("retweets", 0) or 0),
        "tweet_likes":        int(row.get("likes", 0) or 0),
        "tweet_content":      text[:500],
    }


# ── Step 3: Load market data ─────────────────────────────────────────────────
def load_market_data():
    """Returns dict: tweet_date -> {Dow_pct, SP500_pct, NASDAQ_pct}."""
    csv.field_size_limit(10**7)
    market = {}
    with open(FINANCE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts = parse_ts(row.get("timestamp",""))
            if not ts:
                continue
            key = ts.strftime("%Y-%m-%d %H")  # hour-level to handle multiple tweets/day
            d = row.get("Dow_pct","").strip()
            s = row.get("SP500_pct","").strip()
            n = row.get("NASDAQ_pct","").strip()
            if d or s or n:
                market[key] = {
                    "dow_pct":    float(d) if d else None,
                    "sp500_pct":  float(s) if s else None,
                    "nasdaq_pct": float(n) if n else None,
                    "tweet_content_finance": row.get("content","")[:500],
                }
    return market


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    csv.field_size_limit(10**7)

    # 1. Build news index
    news_index = load_news_index()

    # 2. Load market data
    print("Loading market data...")
    market = load_market_data()
    print(f"  {len(market):,} tweet-hours with market data")

    # 3. Process tweets
    print("Processing tweets and aligning with news...")
    master_rows = []
    market_rows = []

    with open(TWEETS_CSV, encoding="utf-8") as f:
        tweets = list(csv.DictReader(f))

    total = len(tweets)
    for i, row in enumerate(tweets):
        if (i+1) % 5000 == 0:
            print(f"  {i+1:,}/{total:,} tweets processed...")

        ts = parse_ts(row.get("timestamp",""))
        if not ts:
            continue

        tf  = get_tweet_features(row)
        nf  = get_news_features(ts, news_index, NEWS_WINDOW_HOURS)
        mhk = ts.strftime("%Y-%m-%d %H")

        combined = {**tf, **nf}

        # Market data (if available)
        mkt = market.get(mhk, {})
        combined["dow_pct"]    = mkt.get("dow_pct")
        combined["sp500_pct"]  = mkt.get("sp500_pct")
        combined["nasdaq_pct"] = mkt.get("nasdaq_pct")
        combined["has_market"] = 1 if mkt else 0

        master_rows.append(combined)

        if mkt:
            market_rows.append(combined)

    # 4. Save
    print(f"\nSaving datasets...")
    df_master = pd.DataFrame(master_rows)
    df_master.to_csv(OUT_MASTER, index=False)
    print(f"  master_dataset.csv : {len(df_master):,} rows, {len(df_master.columns)} columns")

    df_market = pd.DataFrame(market_rows)
    df_market = df_market.dropna(subset=["dow_pct"])
    df_market.to_csv(OUT_MARKET, index=False)
    print(f"  market_dataset.csv : {len(df_market):,} rows with market data")

    # 5. Quick summary
    print("\n── Feature summary (master dataset) ──")
    print(df_master[["tweet_sent","tweet_caps_ratio","tweet_econ",
                      "news_sent_mean","news_econ_intensity","news_trump_pct"]].describe().round(3))


if __name__ == "__main__":
    main()
