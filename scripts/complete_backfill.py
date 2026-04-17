"""
Complete Guardian backfill for remaining years: 2023-2026
"""
from historical_news_fetcher import HistoricalNewsFetcher
from news_database import NewsDatabase
import json

# Load config
with open('news_api_config.json', 'r') as f:
    config = json.load(f)

if not config.get('guardian_key'):
    print("⚠ Guardian API key not found!")
    exit()

# Initialize
fetcher = HistoricalNewsFetcher()
db = NewsDatabase('news_data.db')

print("\n" + "="*70)
print("GUARDIAN: COMPLETING 2023-2026")
print("="*70)

total_articles = 0

# Remaining years
for year in range(2023, 2027):
    # For 2026, only go up to February
    if year == 2026:
        end_month = 2
    else:
        end_month = 12
    
    start_date = f"{year}-01-01"
    end_date = f"{year}-{end_month:02d}-28"
    
    print(f"\n{'='*70}")
    print(f"YEAR {year}: {start_date} to {end_date}")
    print(f"{'='*70}")
    
    articles = fetcher.fetch_guardian_historical(
        'economy OR politics OR finance OR stock market OR trade OR GDP OR business',
        start_date,
        end_date,
        config['guardian_key']
    )
    
    if articles:
        inserted = db.insert_articles_batch(articles)
        total_articles += inserted
        print(f"\n✓ Year {year}: {inserted} new articles stored")
        print(f"Running total: {total_articles:,} articles")
    else:
        print(f"\n⚠ Year {year}: No articles fetched")
    
    stats = db.get_stats()
    print(f"Database now has: {stats['total_articles']:,} total articles")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\nAdded {total_articles:,} articles from 2023-2026")

# Final stats
stats = db.get_stats()
print(f"\n🎉 FINAL DATABASE STATISTICS:")
print(f"  Total articles: {stats['total_articles']:,}")
print(f"  Date range: {stats['earliest_date']} to {stats['latest_date']}")
print(f"\n  Top sources:")
for source, count in stats['by_source'][:10]:
    print(f"    {source}: {count:,}")

print("\n" + "="*70)
print("YOU NOW HAVE 2010-2026 NEWS DATA!")
print("="*70)
