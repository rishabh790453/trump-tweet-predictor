"""
Quick Demo: News Data Collection System
Shows how to use the news fetching and monitoring system
"""
from news_data_fetcher import NewsDataFetcher
from news_database import NewsDatabase
from historical_news_fetcher import HistoricalNewsFetcher
from live_news_monitor import LiveNewsMonitor
import json

def demo_quick_fetch():
    """Demo: Quick news fetch from all sources"""
    print("\n" + "="*70)
    print("DEMO 1: QUICK NEWS FETCH")
    print("="*70 + "\n")
    
    fetcher = NewsDataFetcher('news_api_config.json')
    
    # Fetch from all available sources
    articles = fetcher.fetch_all(query="economy OR stock market")
    
    print(f"\n✓ Fetched {len(articles)} articles")
    
    # Show sample article
    if articles:
        print("\nSample Article:")
        print("-" * 70)
        sample = articles[0]
        print(f"Title: {sample['title']}")
        print(f"Source: {sample['source']}")
        print(f"Published: {sample['published_at']}")
        print(f"URL: {sample['url']}")
        print("-" * 70)
    
    return articles


def demo_store_in_database(articles):
    """Demo: Store articles in database"""
    print("\n" + "="*70)
    print("DEMO 2: STORE IN DATABASE")
    print("="*70 + "\n")
    
    db = NewsDatabase('demo_news.db')
    
    # Store articles
    inserted = db.insert_articles_batch(articles)
    print(f"✓ Stored {inserted} articles in database")
    
    # Get statistics
    stats = db.get_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Total articles: {stats['total_articles']}")
    print(f"  Date range: {stats['earliest_date']} to {stats['latest_date']}")
    print(f"\n  Top sources:")
    for source, count in stats['by_source'][:5]:
        print(f"    {source}: {count}")
    
    return db


def demo_query_database(db):
    """Demo: Query stored articles"""
    print("\n" + "="*70)
    print("DEMO 3: QUERY DATABASE")
    print("="*70 + "\n")
    
    # Get recent articles
    df = db.get_articles(limit=10)
    
    print(f"Recent articles ({len(df)}):")
    print(df[['title', 'source', 'published_at']].to_string(index=False))
    
    return df


def demo_historical_fetch():
    """Demo: Fetch historical data"""
    print("\n" + "="*70)
    print("DEMO 4: HISTORICAL DATA (Sample - 30 days)")
    print("="*70 + "\n")
    
    fetcher = HistoricalNewsFetcher()
    
    # For demo, just fetch 30 days (change to years_back=10 for full history)
    from datetime import datetime, timedelta
    
    # Fetch last 30 days from GDELT
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    articles = fetcher.fetch_gdelt_timeline(
        query="stock market OR economy",
        start_date_str=start_date.strftime('%Y-%m-%d'),
        end_date_str=end_date.strftime('%Y-%m-%d'),
        days_per_chunk=10
    )
    
    print(f"\n✓ Fetched {len(articles)} historical articles")
    
    return articles


def demo_live_monitor_setup():
    """Demo: Live monitoring setup"""
    print("\n" + "="*70)
    print("DEMO 5: LIVE MONITORING SETUP")
    print("="*70 + "\n")
    
    monitor = LiveNewsMonitor(db_path='demo_news.db')
    
    print("Live Monitor Configuration:")
    print(f"  Update interval: {monitor.monitor_config['update_interval_minutes']} minutes")
    print(f"  Query: {monitor.monitor_config['query']}")
    print(f"  Active sources: {sum(monitor.monitor_config['sources'].values())}")
    
    print("\nTo start live monitoring, run:")
    print("  python live_news_monitor.py --mode live")
    
    print("\nTo backfill 10 years of data, run:")
    print("  python live_news_monitor.py --mode backfill --years 10")
    
    return monitor


def demo_export_data(db):
    """Demo: Export data to CSV"""
    print("\n" + "="*70)
    print("DEMO 6: EXPORT DATA")
    print("="*70 + "\n")
    
    output_file = db.export_to_csv('demo_news_export.csv')
    print(f"✓ Exported to {output_file}")
    
    return output_file


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("NEWS DATA COLLECTION SYSTEM - COMPLETE DEMO")
    print("="*70)
    
    # Demo 1: Quick fetch
    articles = demo_quick_fetch()
    
    if not articles:
        print("\n⚠ No articles fetched. This is normal if you haven't set up API keys yet.")
        print("\nTo set up API keys:")
        print("1. Copy news_api_config_template.json to news_api_config.json")
        print("2. Add your API keys (see NEWS_SETUP_GUIDE.md for details)")
        print("3. Run this demo again")
        print("\nNote: GDELT and RSS feeds work without API keys!")
        return
    
    # Demo 2: Store in database
    db = demo_store_in_database(articles)
    
    # Demo 3: Query database
    df = demo_query_database(db)
    
    # Demo 4: Historical fetch (sample)
    hist_articles = demo_historical_fetch()
    if hist_articles:
        db.insert_articles_batch(hist_articles)
        print(f"✓ Historical articles stored in database")
    
    # Demo 5: Live monitor setup
    monitor = demo_live_monitor_setup()
    
    # Demo 6: Export
    export_file = demo_export_data(db)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\nYou now have:")
    print(f"  • Demo database: demo_news.db")
    print(f"  • CSV export: {export_file}")
    print(f"  • {db.get_stats()['total_articles']} total articles stored")
    print("\nNext steps:")
    print("  1. Review NEWS_SETUP_GUIDE.md for full setup instructions")
    print("  2. Configure your API keys in news_api_config.json")
    print("  3. Run: python live_news_monitor.py --mode backfill --years 10")
    print("  4. Run: python live_news_monitor.py --mode live")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
