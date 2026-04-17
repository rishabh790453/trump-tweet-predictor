# News Data Collection System - Setup Guide

## Overview
This system fetches and stores news data from multiple major sources, including:
- **Historical data**: 10+ years of news articles
- **Live monitoring**: Continuous real-time news collection
- **Multiple sources**: NewsAPI, Guardian, NY Times, Alpha Vantage, GDELT, RSS feeds

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_predictor.txt
```

### 2. Get API Keys (All Free!)

#### Required for Maximum Coverage:
- **NewsAPI** (100 requests/day, 1 month history)
  - Sign up: https://newsapi.org/
  - Free tier: Perfect for recent news

- **The Guardian** (Unlimited requests, archive back to 1999!)
  - Sign up: https://open-platform.theguardian.com/access/
  - Free tier: Full access to extensive historical archive

- **NY Times** (4000 requests/day, extensive archive)
  - Sign up: https://developer.nytimes.com/
  - Free tier: Excellent for historical data

- **Alpha Vantage** (News sentiment API)
  - Sign up: https://www.alphavantage.co/support/#api-key
  - Free tier: Good for financial news

#### No API Key Needed:
- **GDELT** - Free, comprehensive news from 1979 to present, updated every 15 minutes
- **RSS Feeds** - Always available, no API key required

### 3. Configure API Keys

Copy the template and add your keys:
```bash
cp news_api_config_template.json news_api_config.json
```

Edit `news_api_config.json`:
```json
{
  "newsapi_key": "YOUR_KEY_HERE",
  "guardian_key": "YOUR_KEY_HERE",
  "nytimes_key": "YOUR_KEY_HERE",
  "alphavantage_key": "YOUR_KEY_HERE"
}
```

**Note**: You can leave any key blank - the system will skip that source and use others.

## Usage

### Option 1: Quick One-Time Fetch
Fetch recent news and store in database:
```bash
python live_news_monitor.py --mode once
```

### Option 2: Backfill Historical Data (10+ Years)
Download 10 years of historical news:
```bash
python live_news_monitor.py --mode backfill --years 10
```

This will fetch from:
- GDELT (free, goes back to 1979)
- The Guardian (if API key configured)
- NY Times (if API key configured)

### Option 3: Start Live Monitoring
Continuously fetch news every 15 minutes:
```bash
python live_news_monitor.py --mode live --interval 15
```

This runs in the background and:
- Fetches from all sources every 15 minutes
- GDELT updates every 15 minutes
- Automatically stores everything in database
- Shows hourly statistics
- Press Ctrl+C to stop

### Option 4: Custom Query
```bash
python live_news_monitor.py --mode once --query "climate change OR renewable energy"
```

### Option 5: Export Data
Export database to CSV:
```bash
python live_news_monitor.py --mode stats --export news_data.csv
```

## Advanced Usage

### Python API

#### Fetch Current News
```python
from news_data_fetcher import NewsDataFetcher
from news_database import NewsDatabase

# Initialize
fetcher = NewsDataFetcher('news_api_config.json')
db = NewsDatabase('news_data.db')

# Fetch from all sources
articles = fetcher.fetch_all(query="economy OR finance")

# Store in database
inserted = db.insert_articles_batch(articles)
print(f"Stored {inserted} articles")
```

#### Fetch Historical Data
```python
from historical_news_fetcher import HistoricalNewsFetcher

fetcher = HistoricalNewsFetcher()

# Fetch 10 years of data
articles = fetcher.fetch_historical_news(
    query="stock market OR economy",
    years_back=10
)

# Save to JSON
import json
with open('historical_data.json', 'w') as f:
    json.dump(articles, f, indent=2)
```

#### Query Database
```python
from news_database import NewsDatabase
import pandas as pd

db = NewsDatabase('news_data.db')

# Get recent articles
df = db.get_articles(
    start_date='2020-01-01',
    end_date='2025-01-01',
    limit=1000
)

# Get statistics
stats = db.get_stats()
print(f"Total articles: {stats['total_articles']}")
print(f"Date range: {stats['earliest_date']} to {stats['latest_date']}")

# Export to CSV
db.export_to_csv('my_news_data.csv', start_date='2020-01-01')
```

#### Live Monitoring Programmatically
```python
from live_news_monitor import LiveNewsMonitor

monitor = LiveNewsMonitor()

# One-time fetch
monitor.fetch_and_store_news()

# Backfill 10 years
monitor.backfill_historical(years=10)

# Start continuous monitoring (blocking call)
monitor.start_monitoring(interval_minutes=15)
```

## Database Schema

The system uses SQLite with these tables:

### news_articles
- `id` - Auto-increment primary key
- `article_id` - Unique article identifier
- `title` - Article title
- `description` - Article description/summary
- `content` - Full article content (when available)
- `source` - News source name
- `author` - Article author
- `url` - Article URL (unique)
- `published_at` - Publication timestamp
- `fetched_at` - When we fetched it
- `category` - Article category
- `sentiment_score` - Sentiment analysis score
- `keywords` - JSON array of keywords
- `image_url` - Article image
- `lang` - Language code

### news_sources
- Tracks all news sources and their fetch status

### fetch_history
- Logs every fetch operation for monitoring

## Data Sources Comparison

| Source | Historical Range | Articles/Request | Rate Limit | Cost |
|--------|-----------------|------------------|------------|------|
| GDELT | 1979 - Present | 250 | None | Free |
| Guardian | 1999 - Present | 50 | Unlimited | Free |
| NY Times | Extensive Archive | 10 | 10/min | Free |
| NewsAPI | 1 month | 100 | 100/day | Free |
| Alpha Vantage | Recent | 50 | 5/min | Free |
| RSS Feeds | Recent | Varies | Varies | Free |

## Configuration Files

### news_api_config.json
Your API keys (keep this private!)

### news_monitor_config.json
Monitoring settings:
- `update_interval_minutes` - How often to fetch (default: 15)
- `query` - Search query
- `sources` - Enable/disable specific sources
- `rss_feeds` - Custom RSS feed URLs

## Tips

1. **Start with GDELT** - It's free, comprehensive, and doesn't require API keys
2. **Get Guardian API Key** - Free unlimited access with great historical data
3. **Use RSS Feeds** - Always available as backup
4. **Run backfill overnight** - 10 years of data takes time
5. **Monitor rate limits** - The system handles this automatically but be aware
6. **Database grows large** - 10 years of data = 100,000+ articles

## Troubleshooting

### "No API key" warnings
- Normal if you haven't configured that source
- System will use other sources automatically

### Rate limit errors
- System includes automatic delays
- Reduce fetch frequency if needed

### Database locked errors
- SQLite can have issues with concurrent writes
- Don't run multiple monitors on same database

### GDELT returns no results
- Try broader search query
- Check date range format

## Examples

### Create a custom news tracker
```python
from live_news_monitor import LiveNewsMonitor

monitor = LiveNewsMonitor()

# Custom query for specific topic
monitor.monitor_config['query'] = 'artificial intelligence OR machine learning'

# Fetch and store
monitor.fetch_and_store_news()

# Export to CSV
monitor.export_data('ai_news.csv')
```

### Analyze news trends
```python
from news_database import NewsDatabase
import pandas as pd

db = NewsDatabase('news_data.db')
df = db.get_articles(start_date='2015-01-01', limit=100000)

# Group by month
df['month'] = pd.to_datetime(df['published_at']).dt.to_period('M')
monthly_counts = df.groupby('month').size()

print(monthly_counts)
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Get API keys (at least 1-2 sources)
3. ✅ Configure `news_api_config.json`
4. ✅ Run one-time fetch to test: `python live_news_monitor.py --mode once`
5. ✅ Backfill historical data: `python live_news_monitor.py --mode backfill --years 10`
6. ✅ Start live monitoring: `python live_news_monitor.py --mode live`

## Support

For issues or questions:
- Check configuration files
- Verify API keys are valid
- Check database file permissions
- Review fetch_history table for errors

Happy news collecting! 📰
