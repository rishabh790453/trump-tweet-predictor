"""
generate_rag.py
===============
RAG pipeline: retrieve similar historical Trump tweets → few-shot prompt Llama 3.

How it works:
  1. Embed input headlines with sentence-transformers
  2. Search FAISS index for top-K most similar historical news contexts
  3. Use those real Trump tweets as few-shot examples in the prompt
  4. Generate with base Llama 3 8B Instruct (no fine-tuning needed)

Run:
  python generate_rag.py --headlines "Trump tariffs China" "Fed holds rates"
  python generate_rag.py --date 2020-11-04
  python generate_rag.py --headlines "..." --k 5 --n 5 --temp 0.8

Setup (first time):
  python build_rag_index.py
"""

import argparse, json, sqlite3
import numpy as np
import faiss
from datetime import datetime, timedelta
from pathlib import Path

INDEX_PATH  = "rag_index.faiss"
DATA_PATH   = "rag_data.jsonl"
NEWS_DB     = "news_data.db"
MODEL_ID    = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MAX_HEADLINES = 10

SYSTEM_PROMPT = (
    "You are Donald Trump. You will be shown examples of real Trump tweets "
    "written in response to similar news, followed by today's headlines. "
    "Write exactly one new tweet in Trump's authentic voice — punchy, opinionated, "
    "capitalized for emphasis, sometimes with exclamation marks. "
    "React directly to the news. Keep it under 280 characters. "
    "Write ONLY the tweet text, nothing else."
)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def load_index():
    if not Path(INDEX_PATH).exists():
        print(f"Index not found. Run: python build_rag_index.py")
        exit(1)
    index = faiss.read_index(INDEX_PATH)
    records = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded index: {index.ntotal:,} vectors, {len(records):,} records")
    return index, records


def retrieve(headlines_str, index, records, encoder, k=5):
    """Find top-k historical (news, tweet) pairs most similar to today's headlines."""
    query_emb = encoder.encode(
        [headlines_str], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    sims, idxs = index.search(query_emb, k)
    results = []
    for sim, idx in zip(sims[0], idxs[0]):
        r = records[idx]
        results.append({
            "news":  r["news"],
            "tweet": r["tweet"],
            "date":  r["date"],
            "sim":   float(sim),
        })
    return results


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(tokenizer, headlines_str, examples):
    """
    Build a few-shot prompt:
      - System: you are Trump
      - User: here are K examples of (news→tweet), now here are today's headlines
      - Assistant: [generation starts here]
    """
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"\n--- Example {i} (similarity: {ex['sim']:.2f}) ---\n"
        examples_text += f"News: {ex['news'][:300]}\n"
        examples_text += f"Tweet: {ex['tweet']}\n"

    user_content = (
        f"Here are {len(examples)} real Trump tweets written when similar news happened:"
        f"\n{examples_text}\n"
        f"---\nNow write a tweet reacting to TODAY's headlines:\n{headlines_str}"
    )

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── Generation ────────────────────────────────────────────────────────────────

def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "mps" if __import__("torch").backends.mps.is_available() else "cpu"
    # float16 causes MPS segfaults on large models — use bfloat16 or float32
    dtype  = __import__("torch").bfloat16

    print(f"Loading {MODEL_ID}...  [{device.upper()}]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print("  Model ready.")
    return model, tokenizer, device


def generate_tweets(model, tokenizer, device, prompt, n=5, temperature=0.8):
    import torch

    input_ids     = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    prompt_len    = input_ids.shape[1]

    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id: terminators.append(eot_id)

    tweets = []
    with torch.no_grad():
        for _ in range(n):
            output = model.generate(
                input_ids,
                attention_mask     = attention_mask,
                max_new_tokens     = 100,
                temperature        = temperature,
                top_p              = 0.9,
                do_sample          = True,
                repetition_penalty = 1.2,
                eos_token_id       = terminators,
                pad_token_id       = tokenizer.eos_token_id,
            )
            new_tokens = output[0][prompt_len:]
            tweet = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            # Clean up — only keep first line if model outputs extra text
            tweet = tweet.split("\n")[0].strip()
            if tweet:
                tweets.append(tweet)
    return tweets


# ── DB headlines ──────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headlines", nargs="+")
    parser.add_argument("--date",      type=str)
    parser.add_argument("--label",     type=str,   default=None, help="output file label")
    parser.add_argument("--k",         type=int,   default=5,  help="# retrieved examples")
    parser.add_argument("--n",         type=int,   default=5,  help="# tweets to generate")
    parser.add_argument("--temp",      type=float, default=0.8)
    args = parser.parse_args()

    print("=" * 60)
    print("TRUMP TWEET GENERATOR — RAG Pipeline")
    print("=" * 60)

    if args.headlines:
        headlines  = args.headlines
        date_label = args.label or "custom"
    elif args.date:
        headlines  = get_headlines_from_db(args.date)
        date_label = args.date
    else:
        print("Provide --headlines or --date")
        return

    if not headlines:
        print("No headlines found.")
        return

    headlines_str = "\n".join(f"- {h}" for h in headlines[:MAX_HEADLINES])
    print(f"\nInput headlines ({len(headlines)}):")
    for h in headlines:
        print(f"  • {h}")

    # Load retrieval components
    from sentence_transformers import SentenceTransformer
    print("\nLoading retriever...")
    encoder = SentenceTransformer(EMBED_MODEL)
    index, records = load_index()

    # Retrieve similar examples
    print(f"\nRetrieving top-{args.k} similar historical tweets...")
    examples = retrieve(headlines_str, index, records, encoder, k=args.k)
    print("\nRetrieved examples:")
    for i, ex in enumerate(examples, 1):
        print(f"  [{i}] sim={ex['sim']:.3f}  date={ex['date']}")
        print(f"       tweet: {ex['tweet'][:80]}...")

    # Load LLM
    print()
    model, tokenizer, device = load_model()

    # Build prompt
    prompt = build_prompt(tokenizer, headlines_str, examples)

    # Generate
    print(f"\nGenerating {args.n} tweets (temp={args.temp})...\n")
    tweets = generate_tweets(model, tokenizer, device, prompt,
                             n=args.n, temperature=args.temp)

    print("=" * 60)
    print("RAG PREDICTED TWEETS:")
    print("=" * 60)
    for i, tweet in enumerate(tweets, 1):
        print(f"\n[{i}] {tweet}")
        print("-" * 60)

    out = {
        "model":       "RAG + Llama3-8B-Instruct",
        "date":        date_label,
        "temperature": args.temp,
        "k_retrieved": args.k,
        "headlines":   headlines,
        "retrieved_examples": examples,
        "predictions": tweets,
    }
    out_file = f"predictions_rag_{date_label}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_file}")


if __name__ == "__main__":
    main()
