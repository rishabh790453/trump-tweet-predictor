"""
Fetch latest news for review before prediction
Uses GDELT (free, no API key needed)
"""

import json
import requests
from datetime import datetime, timedelta

def fetch_gdelt_news(days_back=14):
    """
    Fetch from GDELT v2 (no API key needed, free)
    """
    articles = []

    # GDELT query formats - focus on Trump-relevant topics
    queries = [
        "Trump",
        "economy OR inflation OR GDP",
        "politics OR congress OR election",
        "Israel OR Ukraine OR geopolitics",
        "tariff OR trade war",
        "border OR immigration",
    ]

    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    print(f"\n📰 Fetching from GDELT (no API key needed)...")
    print("=" * 70)

    for query in queries:
        try:
            # GDELT v2 Graph API
            url = "https://api.gdeltproject.org/api/v2/graph/graph"
            params = {
                'query': f'({query}) sourcecountry:US',
                'mode': 'ArtList',
                'maxrecords': 250,
                'format': 'json',
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            for article in data.get('articles', []):
                published = article.get('pubdate', '')

                # Check if after cutoff date
                if published and published >= cutoff:
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('title', '')[:300],
                        'source': article.get('source', 'GDELT'),
                        'url': article.get('url', ''),
                        'published': published,
                    })

            print(f"  ✓ Query '{query}' : {len([a for a in articles if a['source'] == 'GDELT'])} articles")

        except Exception as e:
            print(f"  ✗ Query '{query}' : {str(e)[:50]}")

    # Deduplicate
    seen = set()
    unique = []
    for a in articles:
        if a['title'] not in seen:
            seen.add(a['title'])
            unique.append(a)

    print(f"\n📊 Total unique articles: {len(unique)}")
    print("=" * 70)

    return unique


def categorize_articles(articles):
    """Group articles by topic"""
    import re

    CATEGORIES = {
        "Trump News": r"(trump|donald)",
        "Economy & Jobs": r"(economy|gdp|jobs|unemployment|wage|inflation|interest rate|fed|recession)",
        "Immigration": r"(immigration|border|migrant|visa|daca)",
        "Foreign Policy": r"(russia|ukraine|china|iran|israel|middle east|war|nato|ceasefire)",
        "Politics": r"(congress|senate|representative|dem|gop|election|vote|legislation)",
        "Trade & Tariff": r"(tariff|trade|commerce|export|import)",
        "Other": r".*",
    }

    categorized = {cat: [] for cat in CATEGORIES}

    for article in articles:
        full_text = (article['title'] + " " + article.get('description', '')).lower()

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


def main():
    """Main"""

    print("\n" + "="*70)
    print("📰 NEWS COLLECTION FOR REVIEW")
    print("="*70)

    # Fetch news
    articles = fetch_gdelt_news(days_back=14)

    if not articles:
        print("❌ No articles found!")
        return

    # Categorize
    categorized = categorize_articles(articles)

    # Save for review
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    review_file = f"collected_news_for_review_{timestamp}.json"

    review_data = {
        "generated_at": datetime.now().isoformat(),
        "days_back": 14,
        "total_articles": len(articles),
        "categorized": {
            cat: {
                "count": len(articles),
                "headlines": [
                    {"title": a['title'], "source": a.get('source', 'Unknown'), "published": a.get('published', '')}
                    for a in articles[:10]  # Show top 10 per category
                ]
            }
            for cat, articles in categorized.items()
            if articles
        },
        "all_articles": articles
    }

    with open(review_file, "w") as f:
        json.dump(review_data, f, indent=2)

    print(f"\n✅ Saved to {review_file}")

    # Show summary
    print("\n📑 NEWS BREAKDOWN BY CATEGORY:")
    print("-" * 70)
    for category, articles_list in categorized.items():
        if articles_list:
            print(f"\n{category}: ({len(articles_list)} articles)")
            for i, article in enumerate(articles_list[:5], 1):
                print(f"  {i}. {article['title'][:90]}...")
                print(f"     Source: {article.get('source', 'Unknown')} | {article.get('published', '')}")

    print("\n" + "="*70)
    print(f"📋 FULL LIST SAVED TO: {review_file}")
    print("="*70)
    print("\nReview the file and confirm it contains NEWS HEADLINES ONLY")
    print("(no actual Trump tweets), then use approve_news.py to proceed with predictions")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
