"""
Predict Trump Tweets Based on Latest News
==========================================
Fetches the latest 2 weeks of news and generates predictions using all 4 models
"""

import json
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# NEWS FETCHING
# ────────────────────────────────────────────────────────────────────────────

def fetch_latest_news(days_back=14):
    """Fetch latest news from RSS feeds (most reliable free source)"""
    try:
        import feedparser
    except ImportError:
        print("Installing feedparser...")
        import subprocess
        subprocess.check_call(["pip", "install", "feedparser"])
        import feedparser

    feeds = [
        ('CNN Top Stories', 'http://rss.cnn.com/rss/cnn_topstories.rss'),
        ('Reuters Top News', 'https://www.reuters.com/rssFeed/topNews'),
        ('BBC News', 'http://feeds.bbci.co.uk/news/rss.xml'),
        ('NPR News', 'https://feeds.npr.org/1001/rss.xml'),
        ('Politico', 'https://www.politico.com/rss/politics08.xml'),
        ('CNBC', 'https://www.cnbc.com/id/100003114/device/rss/rss.html'),
        ('Bloomberg Politics', 'https://www.bloomberg.com/politics/feeds/site.xml'),
        ('Fox News', 'http://feeds.foxnews.com/foxnews/latest'),
    ]

    cutoff = datetime.now() - timedelta(days=days_back)
    articles = []

    print(f"\n📰 Fetching news from last {days_back} days...")
    print("=" * 70)

    for source_name, feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            source_title = feed.feed.get('title', source_name)

            for entry in feed.entries[:30]:  # Top 30 per feed
                try:
                    published_str = entry.get('published', entry.get('updated', ''))
                    # Try to parse the date
                    from email.utils import parsedate_to_datetime
                    published = parsedate_to_datetime(published_str) if published_str else datetime.now()

                    if published >= cutoff:
                        articles.append({
                            'title': entry.get('title', ''),
                            'description': entry.get('summary', '')[:300],
                            'source': source_title,
                            'url': entry.get('link', ''),
                            'published': published.isoformat(),
                        })
                except:
                    pass

            print(f"  ✓ {source_name:<25} : {len([a for a in articles if a['source'] == source_title])} articles")
        except Exception as e:
            print(f"  ✗ {source_name:<25} : Error: {str(e)[:50]}")

    # Deduplicate by title
    seen = set()
    unique = []
    for a in articles:
        if a['title'] not in seen:
            seen.add(a['title'])
            unique.append(a)

    print(f"\n📊 Total unique articles: {len(unique)}")
    print("=" * 70)

    return unique


# ────────────────────────────────────────────────────────────────────────────
# NEWS ANALYSIS & CATEGORIZATION
# ────────────────────────────────────────────────────────────────────────────

CATEGORIES = {
    "Economy & Jobs": r"(economy|gdp|jobs|unemployment|wage|inflation|interest rate|fed|recession)",
    "Immigration": r"(immigration|border|migrant|illegal alien|visa|daca)",
    "Foreign Policy": r"(russia|ukraine|china|iran|israel|middle east|war|nato|sanction)",
    "Domestic Politics": r"(congress|senate|representative|dem|gop|election|vote|legislation)",
    "Tech & Innovation": r"(tech|ai|elon|musk|silicon valley|crypto|bitcoin)",
    "Trade & Tariff": r"(tariff|trade|commerce|export|import|china trade)",
    "Energy & Oil": r"(oil|energy|gas|climate|environmental|coal|renewab)",
    "Healthcare": r"(health|medicare|medicaid|obamacare|drug|vaccine)",
    "Crime & Justice": r"(crime|violence|police|law enforcement|jail|prison|court)",
    "Media & Press": r"(media|fake news|cnn|nyt|press)",
}

def categorize_articles(articles):
    """Group articles by topic"""
    categorized = {cat: [] for cat in CATEGORIES}

    for article in articles:
        full_text = (article['title'] + " " + article['description']).lower()

        # Assign to category with highest match
        best_cat = None
        best_matches = 0

        for category, pattern in CATEGORIES.items():
            matches = len(re.findall(pattern, full_text, re.I))
            if matches > best_matches:
                best_matches = matches
                best_cat = category

        if best_cat:
            categorized[best_cat].append(article)

    return categorized


def extract_top_headlines(categorized_articles, max_per_category=3):
    """Extract top headlines from each category"""
    top_headlines = {}

    for category, articles in categorized_articles.items():
        if articles:
            # Take most recent articles from each category
            articles_sorted = sorted(
                articles,
                key=lambda x: x['published'],
                reverse=True
            )
            headlines = [
                f"{a['source']}: {a['title']}"
                for a in articles_sorted[:max_per_category]
            ]
            top_headlines[category] = headlines

    return top_headlines


# ────────────────────────────────────────────────────────────────────────────
# MODEL UTILITIES
# ────────────────────────────────────────────────────────────────────────────

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


SYSTEM_PROMPT = (
    "You are Donald Trump. Given today's news headlines, "
    "write exactly one tweet in Trump's authentic voice — punchy, opinionated, "
    "capitalized for emphasis, sometimes using exclamation marks. "
    "React directly to the news. Keep it under 280 characters."
)

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
# MAIN PREDICTION
# ────────────────────────────────────────────────────────────────────────────

def main():
    """Main prediction pipeline"""

    print("\n" + "="*70)
    print("TRUMP TWEET PREDICTOR - LATEST NEWS")
    print("="*70)

    # Fetch news
    articles = fetch_latest_news(days_back=14)

    if not articles:
        print("❌ No articles found!")
        return

    # Categorize news
    categorized = categorize_articles(articles)
    top_headlines = extract_top_headlines(categorized)

    # Save headlines for review
    print("\n💾 Saving collected news headlines for review...")
    review_file = f"news_for_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    review_data = {
        "generated_at": datetime.now().isoformat(),
        "total_articles": len(articles),
        "categories": top_headlines,
        "all_articles": articles
    }

    with open(review_file, "w") as f:
        json.dump(review_data, f, indent=2)

    print(f"✓ Saved to {review_file}")
    print("\n📋 PLEASE REVIEW THE NEWS HEADLINES BEFORE PROCEEDING!")
    print("=" * 70)

    # Show categories
    print("\n📑 NEWS BREAKDOWN BY CATEGORY:")
    print("-" * 70)
    for category, headlines in top_headlines.items():
        if headlines:
            print(f"\n{category}:")
            for i, headline in enumerate(headlines, 1):
                print(f"  {i}. {headline[:100]}...")

    # Ask for approval
    print("\n" + "="*70)
    response = input("\n✋ Do you approve these headlines? (yes/no): ").strip().lower()

    if response not in ["yes", "y"]:
        print("❌ Aborting predictions. Please review the headlines in " + review_file)
        return

    print("\n✅ Proceeding with predictions...\n")

    # Generate predictions for each category
    results = {}

    for category, headlines in top_headlines.items():
        if not headlines:
            continue

        print(f"\n\n🤖 Generating predictions for: {category}")
        print("-" * 70)

        category_results = {}

        # Markov
        print("  🔗 Markov (Statistical Chain)...", end=" ", flush=True)
        markov_tweets = run_markov(headlines, n=2)
        category_results["markov"] = markov_tweets
        print(f"✓ ({len(markov_tweets)} tweets)")

        # N-gram
        print("  🔤 N-gram (Language Model)...", end=" ", flush=True)
        ngram_tweets = run_ngram(headlines, n=2)
        category_results["ngram"] = ngram_tweets
        print(f"✓ ({len(ngram_tweets)} tweets)")

        # Fine-tuned Llama
        print("  🦙 Fine-tuned Llama 3...", end=" ", flush=True)
        finetune_tweets = run_finetune(headlines, n=2)
        category_results["finetune"] = finetune_tweets
        print(f"✓ ({len(finetune_tweets)} tweets)")

        # RAG
        print("  📚 RAG (Retrieval-Augmented)...", end=" ", flush=True)
        rag_tweets = run_rag(headlines, n=2)
        category_results["rag"] = rag_tweets
        print(f"✓ ({len(rag_tweets)} tweets)")

        results[category] = category_results

        # Show some sample predictions
        print(f"\n  📢 Sample predictions:")
        for model_name, tweets in category_results.items():
            if tweets:
                print(f"\n    {model_name.upper()}:")
                for i, tweet in enumerate(tweets[:1], 1):
                    if tweet:
                        print(f"      \"{tweet}\"")

    # Save results
    print("\n\n" + "="*70)
    print("💾 SAVING RESULTS...")

    output_file = f"predictions/latest_news_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_data = {
        "generated_at": datetime.now().isoformat(),
        "news_articles_count": len(articles),
        "categories": top_headlines,
        "predictions": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved to {output_file}")
    print("\n" + "="*70)
    print("✅ PREDICTION COMPLETE!")
    print("="*70 + "\n")

    return output_data


if __name__ == "__main__":
    main()
