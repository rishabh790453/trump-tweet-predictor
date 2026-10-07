"""
Approve collected news and generate predictions
"""

import json
import re
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def load_news_file(filename):
    """Load the news review file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        sys.exit(1)


SYSTEM_PROMPT = (
    "You are Donald Trump. Given today's news headlines, "
    "write exactly one tweet in Trump's authentic voice — punchy, opinionated, "
    "capitalized for emphasis, sometimes using exclamation marks. "
    "React directly to the news. Keep it under 280 characters."
)

_URL_RE = re.compile(r'https?://\S+|pic\.twitter\.com/\S+|www\.\S+', re.I)


def clean(text):
    """Clean generated text"""
    text = _URL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"!{4,}", "!!!", text)
    text = re.sub(r"\?{4,}", "???", text)
    if len(text) > 280:
        text = text[:280].rsplit(" ", 1)[0]
    return text


# ────────────────────────────────────────────────────────────────────────────
# MARKOV MODEL
# ────────────────────────────────────────────────────────────────────────────

def run_markov(headlines, n=3):
    """Generate tweets using Markov chain"""
    try:
        from markov_predictor import MarkovTweetGenerator
        g = MarkovTweetGenerator(order=2)
        words = re.findall(r"[A-Z][a-z]+|[A-Z]{3,}", " ".join(headlines))[:6]
        keywords = words if words else ["The", "People", "Country"]
        tweets = g.generate_with_topic(keywords, num_tweets=n)
        return [clean(t) for t in tweets]
    except Exception as e:
        print(f"  ⚠️  Markov error: {e}")
        return []


# ────────────────────────────────────────────────────────────────────────────
# N-GRAM MODEL
# ────────────────────────────────────────────────────────────────────────────

def run_ngram(headlines, n=3):
    """Generate tweets using N-gram"""
    try:
        from ngram_predictor import NgramTweetGenerator
        g = NgramTweetGenerator(n=3)
        words = re.findall(r"[A-Z][a-z]+|[A-Z]{3,}", " ".join(headlines))[:6]
        keywords = words if words else ["America", "Great", "People"]
        tweets = g.generate_with_keywords(keywords, num_tweets=n)
        return [clean(t) for t in tweets]
    except Exception as e:
        print(f"  ⚠️  N-gram error: {e}")
        return []


# ────────────────────────────────────────────────────────────────────────────
# FINE-TUNED LLAMA MODEL
# ────────────────────────────────────────────────────────────────────────────

_llama_model = None
_llama_tok = None
_llama_dev = None


def load_llama():
    """Load fine-tuned Llama model"""
    global _llama_model, _llama_tok, _llama_dev
    if _llama_model is not None:
        return
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _llama_dev = "mps" if torch.backends.mps.is_available() else "cpu"
        print("  Loading fine-tuned Llama v3...")
        _llama_tok = AutoTokenizer.from_pretrained("trump_llama_v3")
        _llama_model = AutoModelForCausalLM.from_pretrained(
            "trump_llama_v3", dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(_llama_dev)
        _llama_model.eval()
    except Exception as e:
        print(f"  ⚠️  Could not load Llama: {e}")


def run_finetune(headlines, n=3):
    """Generate tweets using fine-tuned Llama"""
    try:
        import torch
        load_llama()
        if _llama_model is None:
            return []

        hl = "\n".join(f"- {h}" for h in headlines)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Today's news headlines:\n{hl}"},
        ]
        ids = _llama_tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = ids.to(_llama_dev)
        mask = torch.ones_like(ids)
        plen = ids.shape[1]
        terminators = [_llama_tok.eos_token_id,
                       _llama_tok.convert_tokens_to_ids("<|eot_id|>")]
        tweets = []
        with torch.no_grad():
            for _ in range(n):
                out = _llama_model.generate(
                    ids, attention_mask=mask, max_new_tokens=150,
                    temperature=0.85, top_p=0.92, top_k=50, do_sample=True,
                    repetition_penalty=1.2, eos_token_id=terminators,
                    pad_token_id=_llama_tok.eos_token_id,
                )
                t = _llama_tok.decode(out[0][plen:], skip_special_tokens=True).strip()
                tweets.append(clean(t))
        return tweets
    except Exception as e:
        print(f"  ⚠️  Llama error: {e}")
        return []


# ────────────────────────────────────────────────────────────────────────────
# RAG MODEL
# ────────────────────────────────────────────────────────────────────────────

_rag_model = None
_rag_tok = None
_rag_dev = None
_rag_index = None
_rag_data = None
_rag_embed = None


def load_rag():
    """Load RAG components"""
    global _rag_model, _rag_tok, _rag_dev, _rag_index, _rag_data, _rag_embed
    if _rag_model is not None:
        return
    try:
        import torch
        import faiss
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from sentence_transformers import SentenceTransformer
        _rag_dev = "mps" if torch.backends.mps.is_available() else "cpu"
        print("  Loading RAG components...")
        _rag_embed = SentenceTransformer("all-MiniLM-L6-v2")
        _rag_index = faiss.read_index("rag_index.faiss")
        _rag_data = [json.loads(l) for l in open("rag_data.jsonl")]
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        import os
        HF_TOKEN = os.environ.get("HF_TOKEN", "")
        _rag_tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        _rag_model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True, token=HF_TOKEN
        ).to(_rag_dev)
        _rag_model.eval()
    except Exception as e:
        print(f"  ⚠️  Could not load RAG: {e}")


def run_rag(headlines, n=3, k=5):
    """Generate tweets using RAG"""
    try:
        import torch
        import numpy as np
        load_rag()
        if _rag_model is None:
            return []

        query = " ".join(headlines)
        qvec = _rag_embed.encode([query], normalize_embeddings=True).astype("float32")
        _, idxs = _rag_index.search(qvec, k)
        examples = [_rag_data[i] for i in idxs[0] if i < len(_rag_data)]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in examples:
            messages.append({"role": "user", "content": f"Today's news headlines:\n{ex['news']}"})
            messages.append({"role": "assistant", "content": ex["tweet"]})
        hl = "\n".join(f"- {h}" for h in headlines)
        messages.append({"role": "user", "content": f"Today's news headlines:\n{hl}"})

        ids = _rag_tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = ids.to(_rag_dev)
        mask = torch.ones_like(ids)
        plen = ids.shape[1]
        terminators = [_rag_tok.eos_token_id,
                       _rag_tok.convert_tokens_to_ids("<|eot_id|>")]
        tweets = []
        with torch.no_grad():
            for _ in range(n):
                out = _rag_model.generate(
                    ids, attention_mask=mask, max_new_tokens=150,
                    temperature=0.85, top_p=0.92, top_k=50, do_sample=True,
                    repetition_penalty=1.2, eos_token_id=terminators,
                    pad_token_id=_rag_tok.eos_token_id,
                )
                t = _rag_tok.decode(out[0][plen:], skip_special_tokens=True).strip()
                tweets.append(clean(t))
        return tweets
    except Exception as e:
        print(f"  ⚠️  RAG error: {e}")
        return []


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    """Main"""

    if len(sys.argv) < 2:
        print("Usage: python approve_and_predict.py <news_review_file.json>")
        print("Example: python approve_and_predict.py collected_news_for_review_20250601_120000.json")
        sys.exit(1)

    review_file = sys.argv[1]

    print("\n" + "="*70)
    print("🚀 TRUMP TWEET PREDICTION - LATEST NEWS")
    print("="*70)

    # Load news
    print(f"\n📂 Loading news from: {review_file}")
    news_data = load_news_file(review_file)

    articles = news_data.get('all_articles', [])
    print(f"✓ Loaded {len(articles)} articles")

    if not articles:
        print("❌ No articles in file!")
        sys.exit(1)

    # Group articles by simple categories
    categories = {
        "Breaking News": [],
        "Economy": [],
        "Politics": [],
        "Foreign Policy": [],
    }

    for article in articles[:20]:  # Use top 20 articles
        title = article['title'].lower()
        if 'economy' in title or 'inflation' in title or 'jobs' in title:
            categories["Economy"].append(article['title'])
        elif 'trump' in title or 'politics' in title or 'election' in title:
            categories["Politics"].append(article['title'])
        elif 'ukraine' in title or 'israel' in title or 'china' in title or 'russia' in title:
            categories["Foreign Policy"].append(article['title'])
        else:
            categories["Breaking News"].append(article['title'])

    # Show what we're predicting on
    print("\n📰 PROCESSING HEADLINES BY CATEGORY:")
    print("-" * 70)
    for category, headlines in categories.items():
        if headlines:
            print(f"\n{category}: ({len(headlines)} headlines)")
            for i, h in enumerate(headlines[:3], 1):
                print(f"  {i}. {h[:80]}...")

    # Predict
    print("\n\n🤖 GENERATING PREDICTIONS...")
    print("="*70)

    all_results = {}

    for category, headlines in categories.items():
        if not headlines:
            continue

        print(f"\n📌 {category}")
        print("-" * 70)

        category_results = {}

        # Markov
        print("  🔗 Markov...", end=" ", flush=True)
        markov_tweets = run_markov(headlines, n=2)
        category_results["markov"] = markov_tweets
        print(f"✓ ({len(markov_tweets)} tweets)")

        # N-gram
        print("  🔤 N-gram...", end=" ", flush=True)
        ngram_tweets = run_ngram(headlines, n=2)
        category_results["ngram"] = ngram_tweets
        print(f"✓ ({len(ngram_tweets)} tweets)")

        # Fine-tuned Llama
        print("  🦙 Fine-tuned Llama...", end=" ", flush=True)
        finetune_tweets = run_finetune(headlines, n=2)
        category_results["finetune"] = finetune_tweets
        print(f"✓ ({len(finetune_tweets)} tweets)")

        # RAG
        print("  📚 RAG...", end=" ", flush=True)
        rag_tweets = run_rag(headlines, n=2)
        category_results["rag"] = rag_tweets
        print(f"✓ ({len(rag_tweets)} tweets)")

        all_results[category] = {
            "headlines": headlines,
            "predictions": category_results,
        }

        # Show sample
        print(f"\n  📢 Sample Predictions:")
        if markov_tweets:
            print(f"    Markov: \"{markov_tweets[0][:80]}...\"")
        if finetune_tweets:
            print(f"    Llama:  \"{finetune_tweets[0][:80]}...\"")

    # Save results
    print("\n\n" + "="*70)
    print("💾 SAVING RESULTS...")

    output_file = f"predictions/predictions_latest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_data = {
        "generated_at": datetime.now().isoformat(),
        "source_file": review_file,
        "total_articles": len(articles),
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved to {output_file}")
    print("\n" + "="*70)
    print("✅ PREDICTION COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
