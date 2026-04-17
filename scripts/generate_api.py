"""
generate_api.py
===============
Generate Trump tweets via Llama 3 8B on HuggingFace Inference Router.
No local model download or fine-tuning required.

Uses few-shot prompting: samples real Trump tweets from trump_tweets_cleaned.csv
as in-context examples so the model nails his voice immediately.

Usage:
  python generate_api.py --headlines "Trump imposes tariffs on China" "Fed holds rates"
  python generate_api.py --n 6 --temp 0.9
  python generate_api.py  # pulls latest news from NewsAPI automatically

Requires:
  export HF_TOKEN=your_token_here
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

MODEL          = "meta-llama/Meta-Llama-3-8B-Instruct:novita"
HF_BASE_URL    = "https://router.huggingface.co/v1"
TWEETS_CSV     = "trump_tweets_cleaned.csv"
CONFIG_FILE    = "news_api_config.json"
MAX_HEADLINES  = 15
NUM_FEW_SHOT   = 6   # real Trump tweets to include as examples


# ── Load few-shot examples from real tweets ───────────────────────────────────

def load_real_tweets(n=NUM_FEW_SHOT):
    """Sample n real Trump tweets to use as few-shot style examples."""
    if not Path(TWEETS_CSV).exists():
        return []

    import csv
    tweets = []
    with open(TWEETS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = row.get("content", "").strip()
            # Skip retweets, URLs-only, very short/long tweets
            if not content:
                continue
            if content.startswith("RT @"):
                continue
            if re.match(r"^https?://", content):
                continue
            if len(content) < 30 or len(content) > 400:
                continue
            tweets.append(content)

    if not tweets:
        return []

    return random.sample(tweets, min(n, len(tweets)))


def build_system_prompt(few_shot_tweets):
    examples = "\n\n".join(f'"{t}"' for t in few_shot_tweets)
    return f"""You are Donald Trump writing posts on Truth Social or Twitter.

Here are real examples of how Trump writes:

{examples}

When given news headlines, write exactly ONE tweet/post in Trump's authentic voice:
- Punchy and direct
- Uses ALL CAPS for emphasis on key words
- Often ends with exclamation marks!
- Attacks opponents by name when relevant
- Brags about achievements
- Uses phrases like "FAKE NEWS", "WITCH HUNT", "TOTAL DISASTER", "SAD!", "MAKE AMERICA GREAT AGAIN"
- Responds directly to the specific news — do NOT write something generic
- Keep it under 300 characters
- Output ONLY the tweet text, nothing else"""


# ── News fetching ─────────────────────────────────────────────────────────────

def fetch_live_headlines(max_articles=MAX_HEADLINES):
    import requests
    try:
        with open(CONFIG_FILE) as f:
            api_key = json.load(f).get("newsapi_key", "")
    except FileNotFoundError:
        return []

    if not api_key:
        return []

    params = {
        "country":  "us",
        "pageSize": max_articles,
        "apiKey":   api_key,
    }
    resp = requests.get("https://newsapi.org/v2/top-headlines", params=params, timeout=10)
    data = resp.json()
    headlines = []
    for a in data.get("articles", []):
        t = a.get("title", "").strip()
        if t and t != "[Removed]" and len(t) > 15 and " " in t:
            headlines.append(t)
    return headlines


# ── Generation ────────────────────────────────────────────────────────────────

def generate_tweets(headlines, num_tweets=5, temperature=0.85):
    from openai import OpenAI

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("  Run: export HF_TOKEN=your_token_here")
        sys.exit(1)

    client = OpenAI(base_url=HF_BASE_URL, api_key=hf_token)

    few_shot = load_real_tweets(NUM_FEW_SHOT)
    system   = build_system_prompt(few_shot)

    headlines_str = "\n".join(f"- {h}" for h in headlines[:MAX_HEADLINES])
    user_msg      = f"Today's news headlines:\n{headlines_str}\n\nWrite one Trump tweet reacting to this news:"

    tweets = []
    for i in range(num_tweets):
        try:
            resp = client.chat.completions.create(
                model       = MODEL,
                messages    = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = temperature,
                max_tokens  = 200,
            )
            tweet = resp.choices[0].message.content.strip()
            # Strip any surrounding quotes the model might add
            tweet = tweet.strip('"').strip("'").strip()
            if tweet:
                tweets.append(tweet)
        except Exception as e:
            print(f"  [API error on tweet {i+1}]: {e}")

    return tweets


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headlines", nargs="+", help="Headlines to use")
    parser.add_argument("--n",         type=int,   default=5)
    parser.add_argument("--temp",      type=float, default=0.85)
    args = parser.parse_args()

    print("=" * 60)
    print("TRUMP TWEET GENERATOR — Llama 3 8B via HF API")
    print("=" * 60)

    if args.headlines:
        headlines  = args.headlines
        date_label = "custom"
    else:
        print("\nFetching live US headlines from NewsAPI...")
        headlines = fetch_live_headlines()
        date_label = datetime.now().strftime("%Y-%m-%d")
        if not headlines:
            print("No headlines fetched. Use --headlines to pass them manually.")
            sys.exit(1)

    print(f"\nNews context — {len(headlines)} headlines:")
    for h in headlines:
        print(f"  • {h}")

    print(f"\nGenerating {args.n} tweets via Llama 3 8B (temp={args.temp})...\n")
    tweets = generate_tweets(headlines, num_tweets=args.n, temperature=args.temp)

    print("=" * 60)
    print(f"PREDICTED TWEETS  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print("=" * 60)
    for i, tweet in enumerate(tweets, 1):
        print(f"\n[{i}] {tweet}")
        print("-" * 60)

    out = {
        "model":          MODEL,
        "generated_at":   datetime.now().isoformat(),
        "temperature":    args.temp,
        "headlines_used": headlines,
        "predictions":    tweets,
    }
    out_file = f"predictions_api_{date_label}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_file}")


if __name__ == "__main__":
    main()
