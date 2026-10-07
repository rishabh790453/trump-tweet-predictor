"""
Generate Trump Tweet Predictions from Latest News Headlines
Uses news headlines in exact training data format
"""

import json
import re
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def load_news_file(filename):
    """Load the news headlines file"""
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
        print(f"      ⚠️  Error: {str(e)[:60]}")
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
        print(f"      ⚠️  Error: {str(e)[:60]}")
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
        _llama_tok = AutoTokenizer.from_pretrained("trump_llama_v3")
        _llama_model = AutoModelForCausalLM.from_pretrained(
            "trump_llama_v3", dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(_llama_dev)
        _llama_model.eval()
    except Exception as e:
        print(f"      ⚠️  Could not load: {str(e)[:60]}")


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
        print(f"      ⚠️  Error: {str(e)[:60]}")
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
        print(f"      ⚠️  Could not load: {str(e)[:60]}")


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
        print(f"      ⚠️  Error: {str(e)[:60]}")
        return []


# ────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION ENGINE
# ────────────────────────────────────────────────────────────────────────────

def main():
    """Main prediction pipeline"""

    if len(sys.argv) < 2:
        print("Usage: python3 predict_from_news.py <news_file.json>")
        print("Example: python3 predict_from_news.py news_headlines_for_review_20260616_173326.json")
        sys.exit(1)

    news_file = sys.argv[1]

    print("\n" + "="*80)
    print("🤖 TRUMP TWEET PREDICTION ENGINE")
    print("="*80)

    # Load news
    print(f"\n📂 Loading news from: {news_file}")
    news_data = load_news_file(news_file)

    all_headlines = news_data.get('all_headlines', [])
    print(f"✓ Loaded {len(all_headlines)} headlines")

    if not all_headlines:
        print("❌ No headlines in file!")
        sys.exit(1)

    # Show summary
    print(f"\n📑 HEADLINES BY CATEGORY:")
    print("-" * 80)
    for category, data in news_data.get('by_category', {}).items():
        print(f"  • {category}: {data['count']} headlines")

    # Generate predictions by category
    print(f"\n\n" + "="*80)
    print("🚀 GENERATING PREDICTIONS FOR EACH CATEGORY")
    print("="*80)

    all_results = {}

    for category, data in news_data.get('by_category', {}).items():
        headlines = data.get('headlines', [])
        if not headlines:
            continue

        print(f"\n{'─'*80}")
        print(f"📌 {category}")
        print(f"{'─'*80}")

        category_results = {}

        # Show which headlines we're using
        print(f"  Headlines ({len(headlines)}):")
        for h in headlines[:3]:
            print(f"    • {h[:70]}")
        if len(headlines) > 3:
            print(f"    • ... and {len(headlines)-3} more")

        print(f"\n  Models:")

        # Markov
        print(f"    🔗 Markov", end="", flush=True)
        markov_tweets = run_markov(headlines, n=2)
        category_results["markov"] = markov_tweets
        if markov_tweets:
            print(f" ✓ ({len(markov_tweets)} tweets)")
        else:
            print(f" ✗ (no tweets generated)")

        # N-gram
        print(f"    🔤 N-gram", end="", flush=True)
        ngram_tweets = run_ngram(headlines, n=2)
        category_results["ngram"] = ngram_tweets
        if ngram_tweets:
            print(f" ✓ ({len(ngram_tweets)} tweets)")
        else:
            print(f" ✗ (no tweets generated)")

        # Fine-tuned Llama
        print(f"    🦙 Fine-tuned Llama", end="", flush=True)
        finetune_tweets = run_finetune(headlines, n=2)
        category_results["finetune"] = finetune_tweets
        if finetune_tweets:
            print(f" ✓ ({len(finetune_tweets)} tweets)")
        else:
            print(f" ✗ (no tweets generated)")

        # RAG
        print(f"    📚 RAG", end="", flush=True)
        rag_tweets = run_rag(headlines, n=2)
        category_results["rag"] = rag_tweets
        if rag_tweets:
            print(f" ✓ ({len(rag_tweets)} tweets)")
        else:
            print(f" ✗ (no tweets generated)")

        # Show sample predictions
        all_results[category] = {
            "headlines": headlines,
            "predictions": category_results,
        }

        print(f"\n  Sample Tweets:")
        models_with_tweets = [
            ("Markov", markov_tweets),
            ("N-gram", ngram_tweets),
            ("Llama", finetune_tweets),
            ("RAG", rag_tweets),
        ]

        for model_name, tweets in models_with_tweets:
            if tweets:
                tweet_preview = tweets[0][:75]
                print(f"    {model_name:<8} \"{tweet_preview}...\"")

    # Save results
    print(f"\n\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)

    output_file = f"predictions/trump_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_data = {
        "generated_at": datetime.now().isoformat(),
        "source_file": news_file,
        "total_headlines": len(all_headlines),
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Saved to: {output_file}")
    print(f"\n📊 Summary:")
    print(f"   • Total categories analyzed: {len(all_results)}")
    print(f"   • Total headlines processed: {len(all_headlines)}")
    print(f"   • Predictions generated: ~{sum(len(r['predictions']['markov']) for r in all_results.values()) * 4}")

    print("\n" + "="*80)
    print("✅ PREDICTION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
