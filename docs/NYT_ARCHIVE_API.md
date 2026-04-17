# NY Times Archive API - Complete Historical Coverage

## 🎉 What's New

Your system now uses the **NY Times Archive API** which gives you access to **ALL articles back to 1851**!

## Archive API vs Search API

| Feature | Archive API (NEW) | Search API (Old) |
|---------|------------------|------------------|
| **Coverage** | ALL articles | Limited by query |
| **Date Range** | 1851 - Present | Varies |
| **Results per Request** | 2,000-4,000+ | 10 |
| **Pagination** | None needed | Required |
| **Data Size** | ~20MB per month | Small |
| **Best For** | Historical backfill | Searching specific topics |

## How It Works

### Get ALL articles from a single month:
```python
from historical_news_fetcher import HistoricalNewsFetcher

fetcher = HistoricalNewsFetcher()

# Get ALL articles from January 2024
articles = fetcher.fetch_nytimes_archive_month(
    year=2024, 
    month=1, 
    api_key='your_key'
)

# Returns 2,000-4,000 articles typically!
```

### Get ALL articles from a date range:
```python
# Get ALL articles from 2020-2025
articles = fetcher.fetch_nytimes_archive_range(
    '2020-01-01', 
    '2025-12-31', 
    api_key='your_key'
)

# This fetches EVERY month automatically!
# ~150,000+ articles for 5 years!
```

## Example: Test Single Month

Run the test script to see it in action:

```powershell
python test_nyt_archive.py
```

This will fetch ALL NY Times articles from January 2024 (2,000-4,000 articles in one request!)

## Rate Limits

- **10 requests per minute** (same as before)
- Each request gets an entire month
- System automatically adds 6-second delays

## Time Estimates

### For 10 years (2016-2026):
- **120 months** total
- At 6 seconds per month = **12 minutes**
- Result: **~300,000 articles** from NY Times alone!

### For 50 years (1976-2026):
- **600 months** total  
- At 6 seconds per month = **60 minutes**
- Result: **~1.5 million articles**!

## Full Historical Backfill

Now when you run:

```powershell
python live_news_monitor.py --mode backfill --years 10
```

You'll get:
- ✅ **GDELT**: All matching articles (1979-present)
- ✅ **Guardian**: All matching articles (1999-present)
- ✅ **NY Times**: **ALL articles** - complete coverage (not just matching query!)

## API Key Required

Get your free NY Times API key:
1. Visit: https://developer.nytimes.com/
2. Sign up (free)
3. Create an API key
4. Add to `news_api_config.json`:

```json
{
  "nytimes_key": "your_key_here"
}
```

## Data You Get

Each article includes:
- Full headline
- Abstract/description
- Lead paragraph
- Author/byline
- Publication date
- Section/category
- **Keywords** (topics, people, organizations)
- Web URL
- Unique article ID

## Usage in Your System

The Archive API is automatically used in:

1. **Historical backfill:**
   ```powershell
   python live_news_monitor.py --mode backfill --years 10
   ```

2. **Custom date ranges:**
   ```python
   from historical_news_fetcher import HistoricalNewsFetcher
   fetcher = HistoricalNewsFetcher()
   articles = fetcher.fetch_nytimes_archive_range(
       '1990-01-01', 
       '2000-12-31', 
       api_key
   )
   ```

3. **Test single month:**
   ```powershell
   python test_nyt_archive.py
   ```

## Why This Is Awesome

1. **Complete Coverage**: Get EVERYTHING, not just what matches a query
2. **Fast**: One request = entire month (vs hundreds of paginated requests)
3. **Historical**: Goes back to **1851**!
4. **Reliable**: Simple API, no complex filtering needed
5. **Rich Data**: Includes keywords, categories, metadata

## Pro Tips

### Get specific years only
```python
# Just get 2020-2022 (3 years = 36 months = 6 minutes)
articles = fetcher.fetch_nytimes_archive_range(
    '2020-01-01', 
    '2022-12-31', 
    api_key
)
```

### Save JSON monthly archives
```python
# Save each month separately for analysis
articles_jan_2024 = fetcher.fetch_nytimes_archive_month(2024, 1, api_key)

import json
with open('nyt_2024_01.json', 'w') as f:
    json.dump(articles_jan_2024, f, indent=2)
```

### Filter after fetching
```python
# Get ALL articles, then filter
all_articles = fetcher.fetch_nytimes_archive_month(2024, 1, api_key)

# Filter for business section
business = [a for a in all_articles if a['category'] == 'Business']

# Filter for specific keywords
economy = [a for a in all_articles if 'economy' in a.get('keywords', [])]
```

## Next Steps

1. **Test it**: `python test_nyt_archive.py`
2. **Run backfill**: `python live_news_monitor.py --mode backfill --years 10`
3. **Watch your database grow** to hundreds of thousands of articles!

---

**You now have access to the complete NY Times historical archive back to 1851!** 📰🚀
