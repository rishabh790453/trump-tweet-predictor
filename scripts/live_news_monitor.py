"""
Live News Monitoring System
Continuously fetches and stores news data from multiple sources
"""
import time
from datetime import datetime, timedelta
import schedule
import json
import os
from news_data_fetcher import NewsDataFetcher
from news_database import NewsDatabase
from historical_news_fetcher import HistoricalNewsFetcher

class LiveNewsMonitor:
    """
    Continuous news monitoring system
    Fetches news at regular intervals and stores in database
    """
    
    def __init__(self, db_path='news_data.db', config_file='news_api_config.json'):
        """Initialize live news monitor"""
        self.db = NewsDatabase(db_path)
        self.fetcher = NewsDataFetcher(config_file)
        self.historical_fetcher = HistoricalNewsFetcher()
        self.config_file = config_file
        self.is_running = False
        
        # Load or create monitoring config
        self.monitor_config = self.load_monitor_config()
    
    def load_monitor_config(self):
        """Load monitoring configuration"""
        config_file = 'news_monitor_config.json'
        
        default_config = {
            'update_interval_minutes': 15,  # Fetch every 15 minutes
            'query': 'economy OR politics OR finance OR stock market',
            'sources': {
                'newsapi': True,
                'guardian': True,
                'nytimes': True,
                'alphavantage': True,
                'rss': True,
                'gdelt': True
            },
            'auto_start': False,
            'max_articles_per_fetch': 500
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                loaded = json.load(f)
                default_config.update(loaded)
        else:
            # Save default config
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created {config_file} with default settings")
        
        return default_config
    
    def fetch_and_store_news(self, query=None):
        """Fetch news from all sources and store in database"""
        query = query or self.monitor_config['query']
        
        print("\n" + "="*70)
        print(f"LIVE NEWS FETCH - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        try:
            # Fetch from all sources
            articles = self.fetcher.fetch_all(query=query)
            
            # Store in database
            if articles:
                inserted = self.db.insert_articles_batch(articles)
                print(f"\n✓ Stored {inserted} new articles in database")
                self.db.log_fetch('all_sources', inserted, 'success')
            else:
                print("\n⚠ No articles fetched")
                self.db.log_fetch('all_sources', 0, 'no_results')
            
            # Show database stats
            self.print_stats()
            
            return inserted if articles else 0
            
        except Exception as e:
            print(f"\n✗ Error during fetch: {e}")
            self.db.log_fetch('all_sources', 0, 'error', str(e))
            return 0
    
    def fetch_gdelt_live(self, query=None):
        """Fetch latest from GDELT (updated every 15 minutes)"""
        query = query or self.monitor_config['query']
        
        print(f"\nGDELT Live Update - {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Fetch last hour of GDELT data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            start_str = start_time.strftime('%Y%m%d%H%M%S')
            end_str = end_time.strftime('%Y%m%d%H%M%S')
            
            articles = self.historical_fetcher.fetch_gdelt_articles(
                query, start_str, end_str, max_records=250
            )
            
            if articles:
                inserted = self.db.insert_articles_batch(articles)
                print(f"✓ GDELT: {inserted} new articles")
                self.db.log_fetch('gdelt_live', inserted, 'success')
                return inserted
            else:
                print("⚠ GDELT: No new articles")
                return 0
                
        except Exception as e:
            print(f"✗ GDELT error: {e}")
            self.db.log_fetch('gdelt_live', 0, 'error', str(e))
            return 0
    
    def print_stats(self):
        """Print database statistics"""
        stats = self.db.get_stats()
        
        print("\n" + "-"*70)
        print("DATABASE STATISTICS")
        print("-"*70)
        print(f"Total Articles: {stats['total_articles']:,}")
        print(f"Date Range: {stats['earliest_date']} to {stats['latest_date']}")
        print(f"\nArticles by Source:")
        for source, count in stats['by_source'][:10]:
            print(f"  {source}: {count:,}")
        print("-"*70)
    
    def start_monitoring(self, interval_minutes=None):
        """
        Start continuous news monitoring
        
        Args:
            interval_minutes: How often to fetch (default from config)
        """
        interval = interval_minutes or self.monitor_config['update_interval_minutes']
        
        print("\n" + "="*70)
        print("LIVE NEWS MONITORING STARTED")
        print("="*70)
        print(f"Update Interval: Every {interval} minutes")
        print(f"Query: {self.monitor_config['query']}")
        print(f"Database: {self.db.db_path}")
        print(f"Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        self.is_running = True
        
        # Initial fetch
        print("Performing initial fetch...")
        self.fetch_and_store_news()
        
        # Schedule regular fetches
        schedule.every(interval).minutes.do(self.fetch_and_store_news)
        
        # GDELT updates every 15 minutes
        schedule.every(15).minutes.do(self.fetch_gdelt_live)
        
        # Hourly stats report
        schedule.every(1).hours.do(self.print_stats)
        
        # Run scheduler
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("MONITORING STOPPED BY USER")
            print("="*70)
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        self.print_stats()
        schedule.clear()
    
    def backfill_historical(self, years=10):
        """
        One-time backfill of historical data
        
        Args:
            years: How many years back to fetch
        """
        print("\n" + "="*70)
        print(f"BACKFILLING {years} YEARS OF HISTORICAL DATA")
        print("="*70 + "\n")
        
        query = self.monitor_config['query']
        
        # Fetch historical data
        articles = self.historical_fetcher.fetch_historical_news(
            query=query,
            years_back=years,
            config_file=self.config_file
        )
        
        # Store in database
        if articles:
            print(f"\nStoring {len(articles)} articles in database...")
            inserted = self.db.insert_articles_batch(articles)
            print(f"✓ Stored {inserted} new articles (duplicates skipped)")
            self.db.log_fetch('historical_backfill', inserted, 'success')
        
        self.print_stats()
        
        return articles
    
    def export_data(self, output_file='news_export.csv', start_date=None, end_date=None):
        """Export database to CSV"""
        print(f"\nExporting data to {output_file}...")
        self.db.export_to_csv(output_file, start_date, end_date)
        return output_file


def main():
    """Main entry point for live news monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live News Monitoring System')
    parser.add_argument('--mode', choices=['live', 'backfill', 'once', 'stats'], 
                       default='once',
                       help='Operation mode')
    parser.add_argument('--years', type=int, default=10,
                       help='Years of historical data to backfill (default: 10)')
    parser.add_argument('--interval', type=int, default=15,
                       help='Update interval in minutes (default: 15)')
    parser.add_argument('--query', type=str,
                       help='Custom search query')
    parser.add_argument('--export', type=str,
                       help='Export database to CSV file')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = LiveNewsMonitor()
    
    # Set custom query if provided
    if args.query:
        monitor.monitor_config['query'] = args.query
    
    # Execute based on mode
    if args.mode == 'live':
        # Start continuous monitoring
        monitor.start_monitoring(interval_minutes=args.interval)
        
    elif args.mode == 'backfill':
        # Backfill historical data
        monitor.backfill_historical(years=args.years)
        
    elif args.mode == 'once':
        # Single fetch
        monitor.fetch_and_store_news()
        
    elif args.mode == 'stats':
        # Show statistics
        monitor.print_stats()
    
    # Export if requested
    if args.export:
        monitor.export_data(args.export)


if __name__ == "__main__":
    main()
