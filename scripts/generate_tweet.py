"""
generate_tweet.py
=================
Generate predicted Trump tweets from today's news headlines
using the fine-tuned GPT-2 model.

Usage:
  # Interactive mode (fetches live news from DB or manual input)
  python generate_tweet.py

  # Pass a date to use historical news
  python generate_tweet.py --date 2020-11-04

  # Pass headlines directly
  python generate_tweet.py --headlines "Biden wins election" "Stock market surges" "China tariffs"

Requirements:
  pip install torch transformers
"""

import argparse
import sqlite3
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

MODEL_DIR   = "trump_gpt2_model"
NEWS_DB     = "news_data.db"
NEWS_TOKEN  = "<|news|>"
TWEET_TOKEN = "<|tweet|>"
MAX_HEADLINES = 20


def load_model():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model     = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()
    print("  Model loaded.")
    return model, tokenizer


def get_headlines_from_db(date_str):
    """Fetch NYT headlines for a given date (YYYY-MM-DD) from the local DB."""
    conn = sqlite3.connect(NEWS_DB)
    cursor = conn.cursor()
    # Get headlines from the target date and the day before
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        prev = (d - timedelta(days=1)).strftime("%Y-%m-%d")
        cursor.execute(
            """
            SELECT DISTINCT title FROM news_articles
            WHERE source = 'New York Times'
              AND (published_at LIKE ? OR published_at LIKE ?)
              AND title IS NOT NULL AND title != ''
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (f"{date_str}%", f"{prev}%", MAX_HEADLINES),
        )
        headlines = [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()
    return headlines


def get_todays_headlines_from_db():
    """Get the most recent headlines available in the DB."""
    conn = sqlite3.connect(NEWS_DB)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT substr(published_at, 1, 10) as date, COUNT(*) as cnt
            FROM news_articles
            WHERE source = 'New York Times' AND published_at LIKE '20%'
            GROUP BY date ORDER BY date DESC LIMIT 1
            """
        )
        row = cursor.fetchone()
        if row:
            latest_date = row[0]
            print(f"  Most recent news in DB: {latest_date}")
            return get_headlines_from_db(latest_date), latest_date
    finally:
        conn.close()
    return [], None


def build_prompt(headlines):
    """Build the model input prompt."""
    headlines_str = "\n".join(f"- {h}" for h in headlines[:MAX_HEADLINES])
    return f"{NEWS_TOKEN}\n{headlines_str}\n{TWEET_TOKEN}\n"


def generate_tweets(
    model,
    tokenizer,
    headlines,
    num_tweets=5,
    max_new_tokens=200,
    temperature=0.85,
    top_p=0.92,
    top_k=50,
    repetition_penalty=1.3,
):
    import torch

    prompt = build_prompt(headlines)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    tweet_token_id = tokenizer.encode(TWEET_TOKEN, add_special_tokens=False)[0]
    eos_token_id   = tokenizer.eos_token_id

    tweets = []
    with torch.no_grad():
        for i in range(num_tweets):
            output = model.generate(
                input_ids,
                max_new_tokens        = max_new_tokens,
                temperature           = temperature,
                top_p                 = top_p,
                top_k                 = top_k,
                do_sample             = True,
                repetition_penalty    = repetition_penalty,
                eos_token_id          = eos_token_id,
                pad_token_id          = tokenizer.eos_token_id,
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=False)

            # Extract just the tweet part (after <|tweet|>)
            if TWEET_TOKEN in decoded:
                tweet_part = decoded.split(TWEET_TOKEN)[-1]
                # Stop at end token
                if "<|endoftext|>" in tweet_part:
                    tweet_part = tweet_part.split("<|endoftext|>")[0]
                tweet_text = tweet_part.strip()
            else:
                tweet_text = decoded.strip()

            if tweet_text:
                tweets.append(tweet_text)

    return tweets


def interactive_mode():
    """Ask user for headlines manually."""
    print("\nEnter headlines (one per line, blank line to finish):")
    headlines = []
    while True:
        line = input(f"  Headline {len(headlines)+1}: ").strip()
        if not line:
            break
        headlines.append(line)
    return headlines


def main():
    parser = argparse.ArgumentParser(description="Generate Trump tweet predictions")
    parser.add_argument("--date", type=str, help="Date YYYY-MM-DD to use historical news")
    parser.add_argument("--headlines", nargs="+", help="Pass headlines directly")
    parser.add_argument("--n", type=int, default=5, help="Number of tweets to generate")
    parser.add_argument("--temp", type=float, default=0.85, help="Temperature (0.5–1.2)")
    args = parser.parse_args()

    print("=" * 60)
    print("TRUMP TWEET GENERATOR — Fine-tuned GPT-2")
    print("=" * 60)

    if not Path(MODEL_DIR).exists():
        print(f"\nModel not found at {MODEL_DIR}/")
        print("Run these steps first:")
        print("  1. python prepare_finetune_data.py")
        print("  2. pip install torch transformers datasets accelerate")
        print("  3. python finetune_gpt2.py")
        return

    # Get headlines
    if args.headlines:
        headlines = args.headlines
        date_label = "custom"
    elif args.date:
        headlines = get_headlines_from_db(args.date)
        date_label = args.date
    else:
        print("\nFetching most recent headlines from DB...")
        headlines, date_label = get_todays_headlines_from_db()
        if not headlines:
            print("No headlines found in DB. Enter manually:")
            headlines = interactive_mode()
            date_label = "manual"

    if not headlines:
        print("No headlines provided. Exiting.")
        return

    print(f"\nNews context ({date_label}) — {len(headlines)} headlines:")
    for h in headlines[:MAX_HEADLINES]:
        print(f"  • {h}")

    # Load model and generate
    model, tokenizer = load_model()

    print(f"\nGenerating {args.n} tweet predictions (temp={args.temp})...\n")
    tweets = generate_tweets(
        model, tokenizer, headlines,
        num_tweets=args.n,
        temperature=args.temp,
    )

    print("=" * 60)
    print("PREDICTED TWEETS:")
    print("=" * 60)
    for i, tweet in enumerate(tweets, 1):
        print(f"\n[{i}] {tweet}")
        print("-" * 60)

    # Save to file
    out = {
        "date": str(date_label),
        "headlines_used": headlines[:MAX_HEADLINES],
        "temperature": args.temp,
        "predictions": tweets,
    }
    out_file = f"predictions_{date_label}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved predictions → {out_file}")


if __name__ == "__main__":
    main()
