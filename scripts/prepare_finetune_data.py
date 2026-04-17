"""
prepare_finetune_data.py
========================
Builds fine-tuning data by semantically matching Trump tweets to relevant
NYT headlines — instead of blindly pairing every headline from that day.

For each tweet we:
  1. Fetch NYT headlines from the 48h window before the tweet
  2. Encode the tweet + headlines with a sentence embedding model
  3. Keep only the top-K headlines most similar to the tweet (cosine sim)
  4. Discard examples where no headlines clear the similarity threshold
     (i.e. Trump was probably reacting to Fox/Breitbart, not NYT)

This gives the model a real signal: "these headlines → this tweet style."

Output:
  finetune_data.jsonl
  finetune_train.jsonl
  finetune_val.jsonl

Run:
  pip install sentence-transformers
  python prepare_finetune_data.py
  python prepare_finetune_data.py --max-tweets 30000   # for Kaggle run
"""

import argparse
import json
import random
import sqlite3
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
NEWS_DB        = "news_data.db"
TWEETS_CSV     = "trump_tweets_cleaned.csv"
OUT_FULL       = "finetune_data.jsonl"
OUT_TRAIN      = "finetune_train.jsonl"
OUT_VAL        = "finetune_val.jsonl"

MAX_HEADLINES  = 10     # top-K similar headlines to keep per tweet
MIN_HEADLINES  = 3      # skip tweet if fewer than this pass the threshold
MIN_TWEET_LEN  = 30
WINDOW_HOURS   = 48     # look back this many hours for headlines
SIM_THRESHOLD  = 0.20   # minimum cosine similarity to include a headline
TRAIN_RATIO    = 0.90
RANDOM_SEED    = 42
EMBED_MODEL    = "all-MiniLM-L6-v2"   # fast, 384-dim, runs offline
BATCH_SIZE     = 512    # headline encoding batch size

NEWS_TOKEN  = "<|news|>"
TWEET_TOKEN = "<|tweet|>"
END_TOKEN   = "<|endoftext|>"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_tweets(max_tweets=None):
    df = pd.read_csv(TWEETS_CSV)
    df = df.rename(columns={"content": "text"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "text"])
    df = df[df["text"].str.len() >= MIN_TWEET_LEN]
    df = df[~df["text"].str.startswith("RT ")]
    df = df.sort_values("timestamp").reset_index(drop=True)
    if max_tweets:
        # Sample evenly across the timeline rather than truncating
        step = max(1, len(df) // max_tweets)
        df = df.iloc[::step].head(max_tweets).reset_index(drop=True)
    print(f"Loaded {len(df):,} tweets  ({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")
    return df


def load_news(conn):
    """Load all NYT headlines with their dates into a DataFrame."""
    print("Loading NYT headlines from DB...")
    df = pd.read_sql(
        """
        SELECT title, published_at
        FROM news_articles
        WHERE source = 'New York Times'
          AND published_at LIKE '20%'
          AND title IS NOT NULL AND title != ''
        """,
        conn,
    )
    df["date"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["title", "date"])
    print(f"  {len(df):,} unique headlines across {df['date'].nunique():,} dates")
    return df


# ── Semantic matching ─────────────────────────────────────────────────────────

def load_encoder():
    from sentence_transformers import SentenceTransformer
    print(f"Loading sentence encoder ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)
    print("  Encoder ready.")
    return model


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector a and a matrix b (n, d)."""
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_norm @ a_norm


def build_dataset(tweets_df: pd.DataFrame, news_df: pd.DataFrame, encoder) -> list:
    """
    For each tweet, find the semantically closest headlines from the
    preceding WINDOW_HOURS and keep the top MAX_HEADLINES.
    """
    # Group headlines by date for fast lookup
    by_date = news_df.groupby("date")["title"].apply(list).to_dict()

    # Pre-encode all headlines grouped by date to avoid redundant work.
    # We only encode headline sets for dates that actually have tweets nearby.
    tweet_dates = set()
    for ts in tweets_df["timestamp"]:
        d = ts.date()
        tweet_dates.add(d)
        tweet_dates.add((ts - timedelta(days=1)).date())
        tweet_dates.add((ts - timedelta(days=2)).date())

    relevant_dates = tweet_dates & set(by_date.keys())
    print(f"\nPre-encoding headlines for {len(relevant_dates):,} relevant dates...")

    date_embeddings: dict = {}   # date -> (headlines list, np.array of embeddings)
    all_headlines_flat = []
    all_headlines_dates = []
    for d in relevant_dates:
        for h in by_date[d]:
            all_headlines_flat.append(h)
            all_headlines_dates.append(d)

    print(f"  Encoding {len(all_headlines_flat):,} headlines in batches of {BATCH_SIZE}...")
    all_embs = encoder.encode(
        all_headlines_flat,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    # Re-group back by date
    for h, d, emb in zip(all_headlines_flat, all_headlines_dates, all_embs):
        if d not in date_embeddings:
            date_embeddings[d] = ([], [])
        date_embeddings[d][0].append(h)
        date_embeddings[d][1].append(emb)

    # Finalize: stack embedding arrays
    for d in date_embeddings:
        headlines, embs = date_embeddings[d]
        date_embeddings[d] = (headlines, np.stack(embs))

    print("  Done encoding headlines.")

    # Now match each tweet to its most relevant headlines
    examples = []
    skipped_no_news = 0
    skipped_low_sim = 0

    tweet_texts = tweets_df["text"].tolist()
    print(f"\nEncoding {len(tweet_texts):,} tweets...")
    tweet_embs = encoder.encode(
        tweet_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    print("\nMatching tweets to headlines...")
    for i, (_, row) in enumerate(tweets_df.iterrows()):
        ts = row["timestamp"]
        tweet_text = str(row["text"]).strip()
        tweet_emb = tweet_embs[i]

        # Collect headlines from the window
        candidate_headlines = []
        candidate_embs = []
        for offset in range(3):  # today, yesterday, 2 days ago
            d = (ts - timedelta(days=offset)).date()
            if d in date_embeddings:
                hs, es = date_embeddings[d]
                candidate_headlines.extend(hs)
                candidate_embs.append(es)

        if not candidate_headlines:
            skipped_no_news += 1
            continue

        cand_embs_mat = np.concatenate(candidate_embs, axis=0)
        sims = cosine_sim(tweet_emb, cand_embs_mat)

        # Sort by similarity descending
        order = np.argsort(sims)[::-1]
        top_headlines = []
        for idx in order:
            if sims[idx] < SIM_THRESHOLD:
                break
            h = candidate_headlines[idx]
            if h not in top_headlines:  # deduplicate
                top_headlines.append(h)
            if len(top_headlines) >= MAX_HEADLINES:
                break

        if len(top_headlines) < MIN_HEADLINES:
            skipped_low_sim += 1
            continue

        headlines_str = "\n".join(f"- {h}" for h in top_headlines)
        text = f"{NEWS_TOKEN}\n{headlines_str}\n{TWEET_TOKEN}\n{tweet_text}{END_TOKEN}"

        examples.append({
            "text": text,
            "tweet_date": str(ts.date()),
            "tweet_len": len(tweet_text),
            "n_headlines": len(top_headlines),
            "avg_sim": float(np.mean([sims[order[j]] for j in range(len(top_headlines))])),
        })

        if (i + 1) % 5000 == 0:
            print(f"  {i+1:,}/{len(tweets_df):,} tweets processed, {len(examples):,} examples so far")

    print(f"\nBuilt {len(examples):,} training examples")
    print(f"  Skipped (no headlines that day): {skipped_no_news:,}")
    print(f"  Skipped (similarity too low):    {skipped_low_sim:,}")
    if examples:
        avg_sim = np.mean([e["avg_sim"] for e in examples])
        print(f"  Average headline-tweet similarity: {avg_sim:.3f}")
    return examples


# ── Output ────────────────────────────────────────────────────────────────────

def write_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(examples):,} → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tweets", type=int, default=None,
                        help="Limit number of tweets (e.g. 30000 for Kaggle run)")
    parser.add_argument("--threshold", type=float, default=SIM_THRESHOLD,
                        help=f"Min cosine similarity to include a headline (default {SIM_THRESHOLD})")
    args = parser.parse_args()

    print("=" * 60)
    print("PREPARE FINE-TUNE DATA: Semantic News → Trump Tweet")
    print("=" * 60)
    if args.max_tweets:
        print(f"  Max tweets: {args.max_tweets:,}")
    print(f"  Similarity threshold: {args.threshold}")
    print()

    random.seed(RANDOM_SEED)

    tweets_df = load_tweets(max_tweets=args.max_tweets)
    conn      = sqlite3.connect(NEWS_DB)
    news_df   = load_news(conn)
    conn.close()

    encoder  = load_encoder()
    examples = build_dataset(tweets_df, news_df, encoder)

    if not examples:
        print("No examples built — check your data paths.")
        return

    random.shuffle(examples)
    split = int(len(examples) * TRAIN_RATIO)
    train = examples[:split]
    val   = examples[split:]

    print("\nWriting files...")
    write_jsonl(examples, OUT_FULL)
    write_jsonl(train,    OUT_TRAIN)
    write_jsonl(val,      OUT_VAL)

    print("\n" + "=" * 60)
    print("SAMPLE (highest avg similarity):")
    print("=" * 60)
    sample = max(examples[:200], key=lambda x: x["avg_sim"])
    print(f"avg_sim: {sample['avg_sim']:.3f}  |  date: {sample['tweet_date']}")
    print(sample["text"][:600])
    print("...")
    print(f"\nDone. Next: python finetune_llama.py --model llama3 --epochs 3 --max {len(train)}")


if __name__ == "__main__":
    main()
