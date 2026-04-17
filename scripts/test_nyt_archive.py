"""Test NY Times Archive API - Fetch ALL articles from a single month"""
from historical_news_fetcher import HistoricalNewsFetcher
from news_database import NewsDatabase
import json

# Load config
with open('news_api_config.json', 'r') as f:
    config = json.load(f)

# Check if NY Times key is available
if not config.get('nytimes_key'):
    print("⚠ NY Times API key not found in news_api_config.json")
    print("\nTo get a free key:")
    print("1. Visit: https://developer.nytimes.com/")
    print("2. Sign up for free account")
    print("3. Create an API key")
    print("4. Add it to news_api_config.json")
    exit()

# Initialize
fetcher = HistoricalNewsFetcher()
db = NewsDatabase('news_data.db')

# Test: Fetch ALL articles from January 2024
print("="*70)
print("NY TIMES ARCHIVE API TEST")
print("="*70)
print("\nFetching ALL NY Times articles from January 2024...")
print("This demonstrates the Archive API power - gets EVERYTHING!")
print()

articles = fetcher.fetch_nytimes_archive_month(2024, 1, config['nytimes_key'])

if articles:
    print(f"\n✓ Fetched {len(articles)} articles from ONE MONTH!")
    
    # Show sample
    print("\nSample articles:")
    for i, article in enumerate(articles[:5], 1):
        print(f"\n{i}. {article['title'][:80]}")
        print(f"   Category: {article['category']}")
        print(f"   Date: {article['published_at'][:10]}")
    
    # Store in database
    print(f"\nStoring in database...")
    inserted = db.insert_articles_batch(articles)
    print(f"✓ Stored {inserted} new articles")
    
    # Show stats
    stats = db.get_stats()
    print(f"\nTotal articles in database: {stats['total_articles']:,}")
    
    print("\n" + "="*70)
    print("ARCHIVE API is amazing! You can get:")
    print("  • ALL articles from ANY month back to 1851")
    print("  • No pagination needed")
    print("  • Full metadata and keywords")
    print("  • ~2,000-4,000 articles per month typically")
    print("="*70)
else:
    print("\n✗ No articles fetched - check your API key")
