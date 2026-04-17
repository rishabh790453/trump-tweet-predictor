"""
run_pipeline.py
===============
Fetch today's top US news headlines and generate predicted Trump tweets.

Usage:
  python run_pipeline.py                  # fetch live news + generate 5 tweets
  python run_pipeline.py --n 8            # generate 8 tweets
  python run_pipeline.py --temp 1.0       # higher temperature = more creative
  python run_pipeline.py --category all   # include all news categories (default: politics/economy)
  python run_pipeline.py --save-news      # also insert fetched headlines into news_data.db
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

CONFIG_FILE      = "news_api_config.json"
MODEL_DIR_GPT2   = "trump_gpt2_model"
MODEL_DIR_LLAMA  = "trump_llama_model"

# NewsAPI top-headlines categories to pull from
CATEGORIES = {
    "politics":  "Trump OR Congress OR Senate OR White House OR election OR tariff OR Iran",
    "economy":   "economy OR inflation OR stock market OR Federal Reserve OR jobs OR trade",
    "all":       None,  # uses top-headlines endpoint with no query filter
}

# Domains known to produce junk/off-topic results — skip these
BLOCKLIST_KEYWORDS = [
    "beautifulhomes", "viral song", "relationship", "dating", "beautiful homes",
    "recipe", "horoscope", "sports scores", "nfl", "nba", "mlb", "nhl",
]


def is_relevant(title: str) -> bool:
    t = title.lower()
    if any(b in t for b in BLOCKLIST_KEYWORDS):
        return False
    # Must be a real sentence (has a space, reasonable length)
    if len(title) < 15 or " " not in title:
        return False
    return True


# ── News fetching ─────────────────────────────────────────────────────────────

def load_api_key():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f).get("newsapi_key", "")
    except FileNotFoundError:
        return ""


def fetch_headlines(api_key, category="politics", max_articles=20):
    import requests

    base_url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": "us",
        "pageSize": max_articles,
        "apiKey": api_key,
    }

    query = CATEGORIES.get(category)
    if query:
        # Use everything endpoint with query for politics/economy focus
        base_url = "https://newsapi.org/v2/everything"
        params["q"] = query
        params["language"] = "en"
        params["sortBy"] = "publishedAt"
        params["from"] = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        del params["country"]

    resp = requests.get(base_url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data.get('message', 'unknown')}")

    articles = data.get("articles", [])
    headlines = []
    for a in articles:
        title = a.get("title", "").strip()
        # Filter out junk titles
        if title and title != "[Removed]" and is_relevant(title):
            headlines.append({
                "title": title,
                "source": a.get("source", {}).get("name", ""),
                "published_at": a.get("publishedAt", ""),
                "url": a.get("url", ""),
            })

    return headlines


def save_to_db(articles):
    import sqlite3, hashlib
    conn = sqlite3.connect("news_data.db")
    inserted = 0
    for a in articles:
        uid = hashlib.md5(a["url"].encode()).hexdigest()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO news_articles
                   (article_id, title, source, url, published_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (uid, a["title"], a["source"], a["url"], a["published_at"]),
            )
            inserted += conn.execute("SELECT changes()").fetchone()[0]
        except Exception:
            pass
    conn.commit()
    conn.close()
    return inserted


# ── Tweet generation ──────────────────────────────────────────────────────────

def load_model():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model     = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer


def generate_tweets(model, tokenizer, headlines, num_tweets=5, temperature=0.85):
    import torch

    NEWS_TOKEN  = "<|news|>"
    TWEET_TOKEN = "<|tweet|>"

    headlines_str = "\n".join(f"- {h}" for h in headlines[:20])
    prompt = f"{NEWS_TOKEN}\n{headlines_str}\n{TWEET_TOKEN}\n"

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    eos_id    = tokenizer.eos_token_id

    tweets = []
    with torch.no_grad():
        for _ in range(num_tweets):
            output = model.generate(
                input_ids,
                max_new_tokens     = 200,
                temperature        = temperature,
                top_p              = 0.92,
                top_k              = 50,
                do_sample          = True,
                repetition_penalty = 1.3,
                eos_token_id       = eos_id,
                pad_token_id       = eos_id,
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=False)
            if TWEET_TOKEN in decoded:
                tweet = decoded.split(TWEET_TOKEN)[-1]
                if "<|endoftext|>" in tweet:
                    tweet = tweet.split("<|endoftext|>")[0]
                tweet = tweet.strip()
            else:
                tweet = decoded.strip()

            if tweet:
                tweets.append(tweet)

    return tweets


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch news → generate Trump tweets")
    parser.add_argument("--n",         type=int,   default=5,          help="Number of tweets to generate")
    parser.add_argument("--temp",      type=float, default=0.85,       help="Generation temperature (0.5–1.2)")
    parser.add_argument("--category",  type=str,   default="politics", choices=list(CATEGORIES), help="News category filter")
    parser.add_argument("--max-news",  type=int,   default=15,         help="Max headlines to fetch")
    parser.add_argument("--save-news", action="store_true",            help="Save fetched headlines to news_data.db")
    parser.add_argument("--model",     type=str,   default="auto",     choices=["auto", "api", "llama", "gpt2"],
                        help="Model: api=Llama3 via HF API, llama=local LoRA, gpt2=original, auto=best available")
    args = parser.parse_args()

    print("=" * 60)
    print("TRUMP TWEET PIPELINE — Live News → GPT-2")
    print("=" * 60)

    # Pick model
    import os as _os
    has_hf_token  = bool(_os.environ.get("HF_TOKEN"))
    has_llama_dir = Path(MODEL_DIR_LLAMA).exists()

    if args.model == "auto":
        if has_hf_token:
            args.model = "api"
        elif has_llama_dir:
            args.model = "llama"
        else:
            args.model = "gpt2"

    model_labels = {"api": "Llama 3 8B (HF API)", "llama": "Llama/Mistral (local LoRA)", "gpt2": "GPT-2"}
    print(f"Using model: {model_labels[args.model]}")

    # 1. Fetch news
    api_key = load_api_key()
    if not api_key:
        print("No NewsAPI key found in news_api_config.json. Exiting.")
        sys.exit(1)

    print(f"\nFetching top US news [{args.category}]...")
    try:
        articles  = fetch_headlines(api_key, category=args.category, max_articles=args.max_news)
        headlines = [a["title"] for a in articles]
    except Exception as e:
        print(f"Failed to fetch news: {e}")
        sys.exit(1)

    if not headlines:
        print("No headlines retrieved. Exiting.")
        sys.exit(1)

    print(f"  Got {len(headlines)} headlines:\n")
    for h in headlines:
        print(f"  • {h}")

    if args.save_news:
        n = save_to_db(articles)
        print(f"\n  Saved {n} new articles to news_data.db")

    # 2. Load model + generate
    if args.model == "api":
        from generate_api import generate_tweets as gen_api
        print(f"\nCalling Llama 3 8B via HF API...")
        tweets = gen_api(headlines, num_tweets=args.n, temperature=args.temp)
    elif args.model == "llama":
        print(f"\nLoading local Llama/Mistral model...")
        from generate_llama import load_model as load_llama, generate_tweets as gen_llama
        model, tokenizer, device = load_llama()
        tweets = gen_llama(model, tokenizer, device, headlines,
                           num_tweets=args.n, temperature=args.temp)
    else:
        print(f"\nLoading GPT-2 model...")
        model, tokenizer = load_model()
        tweets = generate_tweets(model, tokenizer, headlines,
                                 num_tweets=args.n, temperature=args.temp)

    print("=" * 60)
    print(f"PREDICTED TWEETS  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print("=" * 60)
    for i, tweet in enumerate(tweets, 1):
        print(f"\n[{i}] {tweet}")
        print("-" * 60)

    # 3. Save output
    out = {
        "generated_at": datetime.now().isoformat(),
        "category": args.category,
        "temperature": args.temp,
        "headlines_used": headlines,
        "predictions": tweets,
    }
    out_file = f"predictions_{datetime.now().strftime('%Y-%m-%d_%H%M')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_file}")


if __name__ == "__main__":
    main()
