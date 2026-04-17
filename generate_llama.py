"""
generate_llama.py
=================
Generate Trump tweets using the fine-tuned Llama/Mistral model.
Drop-in replacement for generate_tweet.py with much better quality.

Usage:
  python generate_llama.py --headlines "Trump imposes new tariffs" "Fed cuts rates"
  python generate_llama.py --date 2020-11-04
  python generate_llama.py --n 8 --temp 0.9
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

MODEL_DIR   = "trump_llama_v3"
NEWS_DB     = "news_data.db"
MAX_HEADLINES = 15

SYSTEM_PROMPT = (
    "You are Donald Trump. Given today's news headlines, "
    "write exactly one tweet in Trump's authentic voice — punchy, opinionated, "
    "capitalized for emphasis, sometimes using exclamation marks. "
    "React directly to the news. Keep it under 280 characters."
)


def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype  = torch.bfloat16

    print(f"Loading model from {MODEL_DIR}...  [{device.upper()}]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print("  Model loaded.")
    return model, tokenizer, device


def get_headlines_from_db(date_str):
    conn   = sqlite3.connect(NEWS_DB)
    cursor = conn.cursor()
    try:
        d    = datetime.strptime(date_str, "%Y-%m-%d")
        prev = (d - timedelta(days=1)).strftime("%Y-%m-%d")
        cursor.execute(
            """SELECT DISTINCT title FROM news_articles
               WHERE source = 'New York Times'
                 AND (published_at LIKE ? OR published_at LIKE ?)
                 AND title IS NOT NULL AND title != ''
               ORDER BY published_at DESC LIMIT ?""",
            (f"{date_str}%", f"{prev}%", MAX_HEADLINES),
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def generate_tweets(model, tokenizer, device, headlines, num_tweets=5, temperature=0.85):
    import torch

    headlines_str = "\n".join(f"- {h}" for h in headlines[:MAX_HEADLINES])
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Today's news headlines:\n{headlines_str}"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    # apply_chat_template may return a BatchEncoding or plain tensor
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids.input_ids
    input_ids = input_ids.to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    prompt_len = input_ids.shape[1]

    # Llama 3 needs both eos_token and <|eot_id|> as stop tokens
    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None:
        terminators.append(eot_id)

    tweets = []
    with torch.no_grad():
        for _ in range(num_tweets):
            output = model.generate(
                input_ids,
                attention_mask     = attention_mask,
                max_new_tokens     = 150,
                temperature        = temperature,
                top_p              = 0.92,
                top_k              = 50,
                do_sample          = True,
                repetition_penalty = 1.2,
                eos_token_id       = terminators,
                pad_token_id       = tokenizer.eos_token_id,
            )
            # Only decode the newly generated tokens
            new_tokens = output[0][prompt_len:]
            tweet = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            # Post-processing: strip URLs, collapse excess punctuation, enforce 280 chars
            import re as _re
            tweet = _re.sub(r'https?://\S+|pic\.twitter\.com/\S+|www\.\S+', '', tweet)
            tweet = _re.sub(r'\s+', ' ', tweet).strip()
            tweet = _re.sub(r'!{4,}', '!!!', tweet)   # max 3 consecutive !
            tweet = _re.sub(r'\?{4,}', '???', tweet)
            tweet = tweet[:280].rsplit(' ', 1)[0] if len(tweet) > 280 else tweet
            if tweet:
                tweets.append(tweet)

    return tweets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headlines", nargs="+", help="Pass headlines directly")
    parser.add_argument("--date",      type=str,   help="YYYY-MM-DD to use DB headlines")
    parser.add_argument("--n",         type=int,   default=5)
    parser.add_argument("--temp",      type=float, default=0.85)
    args = parser.parse_args()

    print("=" * 60)
    print("TRUMP TWEET GENERATOR — Fine-tuned Llama/Mistral")
    print("=" * 60)

    if not Path(MODEL_DIR).exists():
        print(f"\nModel not found at {MODEL_DIR}/")
        print("Run: python finetune_llama.py")
        return

    if args.headlines:
        headlines  = args.headlines
        date_label = "custom"
    elif args.date:
        headlines  = get_headlines_from_db(args.date)
        date_label = args.date
    else:
        print("\nNo headlines provided. Use --headlines or --date.")
        return

    if not headlines:
        print("No headlines found.")
        return

    print(f"\nNews context ({date_label}) — {len(headlines)} headlines:")
    for h in headlines:
        print(f"  • {h}")

    model, tokenizer, device = load_model()

    print(f"\nGenerating {args.n} tweets (temp={args.temp})...\n")
    tweets = generate_tweets(model, tokenizer, device, headlines,
                             num_tweets=args.n, temperature=args.temp)

    print("=" * 60)
    print("PREDICTED TWEETS:")
    print("=" * 60)
    for i, tweet in enumerate(tweets, 1):
        print(f"\n[{i}] {tweet}")
        print("-" * 60)

    out = {
        "model":          MODEL_DIR,
        "date":           str(date_label),
        "temperature":    args.temp,
        "headlines_used": headlines,
        "predictions":    tweets,
    }
    out_file = f"predictions_llama_{date_label}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved predictions → {out_file}")


if __name__ == "__main__":
    main()
