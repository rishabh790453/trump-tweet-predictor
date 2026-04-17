# News Data Collection System

## 🚀 Complete System for Historical & Live News Data

This system provides comprehensive news data collection from major sources with:
- ✅ **10+ years of historical data** (back to 1979 with GDELT)
- ✅ **Live continuous monitoring** (updates every 15 minutes)
- ✅ **Multiple major sources** (NewsAPI, Guardian, NY Times, Alpha Vantage, GDELT, RSS)
- ✅ **SQLite database storage** with full text search
- ✅ **Easy CSV export** for analysis

## 📦 Files Overview

| File | Purpose |
|------|---------|
| `news_data_fetcher.py` | Fetch news from multiple sources (NewsAPI, Guardian, NY Times, Alpha Vantage, RSS) |
| `historical_news_fetcher.py` | Fetch 10+ years of historical data (GDELT, Guardian, NY Times) |
| `live_news_monitor.py` | Continuous monitoring system - **Main entry point** |
| `news_database.py` | SQLite database management |
| `demo_news_system.py` | Complete demo showing all features |
| `NEWS_SETUP_GUIDE.md` | **Full setup and usage guide** |
| `news_api_config.json` | Your API keys (create from template) |
| `news_monitor_config.json` | Monitoring configuration |

## ⚡ Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements_predictor.txt
```

### 2. Get FREE API Keys
See [NEWS_SETUP_GUIDE.md](NEWS_SETUP_GUIDE.md) for links. All sources offer free tiers!

**Start immediately without API keys:**
- GDELT (1979-present, free, no key needed)
- RSS feeds (always available)

### 3. Run It!

**Option A: Quick test** (works without API keys)
```bash
python live_news_monitor.py --mode once
```

**Option B: Get 10 years of historical data**
```bash
python live_news_monitor.py --mode backfill --years 10
```

**Option C: Start live monitoring**
```bash
python live_news_monitor.py --mode live
```

## 📊 Data Sources

| Source | Free Tier | Historical Range | Rate Limit |
|--------|-----------|------------------|------------|
| **GDELT** | ✅ Unlimited | **1979 - Present** | None |
| **Guardian** | ✅ Unlimited | 1999 - Present | None |
| **NY Times** | ✅ 4000/day | Extensive Archive | 10/min |
| **NewsAPI** | ✅ 100/day | 1 month | 100/day |
| **Alpha Vantage** | ✅ 500/day | Recent | 5/min |
| **RSS Feeds** | ✅ Always Free | Recent | Varies |

## 💡 Common Use Cases

### Backfill 10 years of data
```bash
python live_news_monitor.py --mode backfill --years 10
```

### Monitor specific topics
```bash
python live_news_monitor.py --mode live --query "climate change OR renewable energy"
```

### Export to CSV
```bash
python live_news_monitor.py --mode stats --export my_news_data.csv
```

### Run demo
```bash
python demo_news_system.py
```

## 🔧 Python API Usage

```python
from live_news_monitor import LiveNewsMonitor

# Initialize
monitor = LiveNewsMonitor()

# Fetch once
monitor.fetch_and_store_news()

# Backfill historical
monitor.backfill_historical(years=10)

# Export to CSV
monitor.export_data('news_data.csv')

# Get stats
stats = monitor.db.get_stats()
print(f"Total articles: {stats['total_articles']}")
```

## 📖 Full Documentation

See [NEWS_SETUP_GUIDE.md](NEWS_SETUP_GUIDE.md) for:
- Detailed setup instructions
- API key registration links
- Advanced usage examples
- Database schema
- Troubleshooting
- Python API reference

## 🎯 What You Get

After backfilling 10 years:
- **100,000+ news articles** from major sources
- **Full text content** (where available)
- **Metadata**: source, author, publish date, categories
- **Sentiment scores** (from Alpha Vantage)
- **SQLite database** for fast queries
- **CSV export** for analysis

## 🔄 Live Monitoring

Once started, the system:
- ✅ Fetches from all sources every 15 minutes
- ✅ GDELT updates every 15 minutes (most current news)
- ✅ Automatically filters duplicates
- ✅ Shows hourly statistics
- ✅ Logs all operations
- ✅ Runs 24/7 in background

## 🎓 Examples

Check out:
- `demo_news_system.py` - Complete working demo
- `NEWS_SETUP_GUIDE.md` - Full examples and tutorials

## 📝 Configuration

Edit `news_monitor_config.json` to:
- Change update frequency
- Customize search queries
- Enable/disable sources
- Add custom RSS feeds

## 🆘 Need Help?

1. Read [NEWS_SETUP_GUIDE.md](NEWS_SETUP_GUIDE.md)
2. Run demo: `python demo_news_system.py`
3. Check database: `python live_news_monitor.py --mode stats`

## 🎉 You're Ready!

Start with:
```bash
python demo_news_system.py
```

Then read NEWS_SETUP_GUIDE.md for the complete guide.

Happy news collecting! 📰
