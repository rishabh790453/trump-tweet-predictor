"""Quick test of Guardian historical data"""
from historical_news_fetcher import HistoricalNewsFetcher
from news_database import NewsDatabase
import json

# Load config
with open('news_api_config.json', 'r') as f:
    config = json.load(f)

# Initialize
fetcher = HistoricalNewsFetcher()
db = NewsDatabase('news_data.db')

# Test fetch for 2020
print("Fetching Guardian historical data for 2020...")
articles = fetcher.fetch_guardian_historical(
    'economy OR finance', 
    '2020-01-01', 
    '2020-12-31', 
    config['guardian_key']
)

# Store in database
if articles:
    inserted = db.insert_articles_batch(articles)
    print(f'\n✓ Stored {inserted} historical articles from 2020')
    
    stats = db.get_stats()
    print(f'Total in database: {stats["total_articles"]:,}')
else:
    print("No articles fetched")
