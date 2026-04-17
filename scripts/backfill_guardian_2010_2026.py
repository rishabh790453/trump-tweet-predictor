"""
Backfill Guardian Historical Data: 2010-2026
Guardian has unlimited API requests and extensive archive!
"""
from historical_news_fetcher import HistoricalNewsFetcher
from news_database import NewsDatabase
import json
from datetime import datetime, timedelta

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
print("GUARDIAN HISTORICAL BACKFILL: 2010-2026")
print("="*70)
print("\nThis will fetch ALL Guardian articles from 2010 to today")
print("Guardian API: Unlimited requests, excellent historical coverage")
print("="*70 + "\n")

# We'll break it into yearly chunks to track progress better
start_year = 2010
end_year = 2026
current_month = 2  # February 2026

total_articles = 0

for year in range(start_year, end_year + 1):
    # For the current year (2026), only go up to current month
    if year == end_year:
        end_month = current_month
    else:
        end_month = 12
    
    # Fetch this year
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
    
    # Show current stats
    stats = db.get_stats()
    print(f"Database now has: {stats['total_articles']:,} total articles")

print("\n" + "="*70)
print("BACKFILL COMPLETE!")
print("="*70)
print(f"\nTotal Guardian articles added: {total_articles:,}")

# Final stats
stats = db.get_stats()
print(f"\nFinal Database Statistics:")
print(f"  Total articles: {stats['total_articles']:,}")
print(f"  Date range: {stats['earliest_date']} to {stats['latest_date']}")
print(f"\n  Top sources:")
for source, count in stats['by_source'][:10]:
    print(f"    {source}: {count:,}")

print("\n" + "="*70)
print("You now have 16 years of news data from The Guardian!")
print("="*70)
