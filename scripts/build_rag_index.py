"""
build_rag_index.py
==================
One-time setup: builds a FAISS vector index over all (news, tweet) pairs.

Run once before using generate_rag.py:
  python build_rag_index.py

Output:
  rag_index.faiss  — FAISS index of news embeddings
  rag_data.jsonl   — parallel file: news + tweet for each indexed vector
"""

import json, re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH   = "finetune_data.jsonl"
INDEX_PATH  = "rag_index.faiss"
DATA_OUT    = "rag_data.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE  = 512
NEWS_TOKEN  = "<|news|>"
TWEET_TOKEN = "<|tweet|>"
EOS_TOKEN   = "<|endoftext|>"


def parse_example(text):
    if NEWS_TOKEN not in text or TWEET_TOKEN not in text:
        return None, None
    news  = text.split(NEWS_TOKEN)[1].split(TWEET_TOKEN)[0].strip()
    tweet = text.split(TWEET_TOKEN)[1].replace(EOS_TOKEN, "").strip()
    if not news or not tweet: return None, None
    if tweet.startswith("RT @") or re.match(r"^https?://", tweet): return None, None
    if len(tweet) < 20 or len(tweet) > 600: return None, None
    return news, tweet


def main():
    print("=" * 60)
    print("Building RAG Index")
    print("=" * 60)

    print(f"\nLoading {DATA_PATH}...")
    records = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                news, tweet = parse_example(obj.get("text", ""))
                if news is None: continue
                records.append({
                    "news":  news,
                    "tweet": tweet,
                    "date":  obj.get("tweet_date", ""),
                    "sim":   obj.get("avg_sim", 0.0),
                })
            except: continue
    print(f"  Loaded {len(records):,} examples")

    print(f"\nEncoding with {EMBED_MODEL}...")
    encoder    = SentenceTransformer(EMBED_MODEL)
    embeddings = encoder.encode(
        [r["news"] for r in records],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine via inner product
    )

    print("\nBuilding FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, INDEX_PATH)
    print(f"  Saved {index.ntotal:,} vectors → {INDEX_PATH}")

    with open(DATA_OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved → {DATA_OUT}")
    print("\nDone. Run: python generate_rag.py --headlines '...'")


if __name__ == "__main__":
    main()
