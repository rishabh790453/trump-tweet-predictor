"""
Historical News Data Fetcher
Fetch news data going back 10+ years from GDELT and other sources
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
from typing import List, Dict
import hashlib

class HistoricalNewsFetcher:
    """
    Fetch historical news data from multiple sources
    Primary source: GDELT Project (free, goes back to 1979)
    """
    
    def __init__(self):
        self.gdelt_base_url = "https://api.gdeltproject.org/api/v2"
    
    def generate_article_id(self, url, title):
        """Generate unique article ID"""
        unique_str = f"{url}{title}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    # ============ GDELT Project (Best for Historical Data) ============
    def fetch_gdelt_articles(self, query, start_date, end_date, max_records=250):
        """
        Fetch articles from GDELT Project
        
        GDELT is a free, open platform with news from 1979 to present
        Updated every 15 minutes with global news coverage
        
        Args:
            query: Search keywords
            start_date: Start date (YYYYMMDDHHMMSS format)
            end_date: End date (YYYYMMDDHHMMSS format)
            max_records: Max articles to fetch (default 250, max 250 per request)
        """
        url = f"{self.gdelt_base_url}/doc/doc"
        
        params = {
            'query': query,
            'mode': 'artlist',
            'maxrecords': min(max_records, 250),
            'format': 'json',
            'startdatetime': start_date,
            'enddatetime': end_date
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # GDELT returns HTML table, we need to parse it
            data = response.json()
            
            articles = []
            if 'articles' in data:
                for article in data['articles']:
                    articles.append({
                        'article_id': self.generate_article_id(
                            article.get('url', ''),
                            article.get('title', '')
                        ),
                        'title': article.get('title', ''),
                        'description': article.get('seendate', ''),
                        'content': '',  # GDELT doesn't provide full content
                        'source': article.get('domain', 'GDELT'),
                        'author': '',
                        'url': article.get('url', ''),
                        'published_at': self.parse_gdelt_date(article.get('seendate', '')),
                        'image_url': article.get('socialimage', ''),
                        'category': 'historical',
                        'lang': article.get('language', 'en')
                    })
            
            return articles
            
        except Exception as e:
            print(f"GDELT error: {e}")
            return []
    
    def parse_gdelt_date(self, gdelt_date):
        """Convert GDELT date format to ISO format"""
        try:
            if len(gdelt_date) == 14:  # YYYYMMDDHHMMSS
                dt = datetime.strptime(gdelt_date, '%Y%m%d%H%M%S')
                return dt.isoformat()
            return gdelt_date
        except:
            return gdelt_date
    
    def fetch_gdelt_timeline(self, query, start_date_str, end_date_str, 
                            days_per_chunk=30):
        """
        Fetch GDELT data over a long time period by chunking
        
        Args:
            query: Search query
            start_date_str: Start date as string 'YYYY-MM-DD'
            end_date_str: End date as string 'YYYY-MM-DD'
            days_per_chunk: Days per request (GDELT limits per request)
        """
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        all_articles = []
        current_date = start_date
        
        total_days = (end_date - start_date).days
        print(f"\nFetching {total_days} days of historical data from GDELT...")
        print(f"Date range: {start_date_str} to {end_date_str}")
        print(f"Query: {query}\n")
        
        chunk_num = 0
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=days_per_chunk), end_date)
            
            # Format dates for GDELT
            start_str = current_date.strftime('%Y%m%d%H%M%S')
            end_str = chunk_end.strftime('%Y%m%d%H%M%S')
            
            chunk_num += 1
            print(f"Chunk {chunk_num}: {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}", end=" ")
            
            articles = self.fetch_gdelt_articles(query, start_str, end_str)
            all_articles.extend(articles)
            
            print(f"→ {len(articles)} articles")
            
            # Rate limiting
            time.sleep(2)
            
            current_date = chunk_end
        
        print(f"\n✓ Total: {len(all_articles)} articles fetched from GDELT")
        return all_articles
    
    # ============ Guardian Historical Archive ============
    def fetch_guardian_historical(self, query, start_date_str, end_date_str, api_key):
        """
        Fetch historical data from Guardian (goes back to 1999)
        
        Args:
            query: Search query
            start_date_str: 'YYYY-MM-DD'
            end_date_str: 'YYYY-MM-DD'
            api_key: Guardian API key
        """
        url = "https://content.guardianapis.com/search"
        
        params = {
            'q': query,
            'from-date': start_date_str,
            'to-date': end_date_str,
            'page-size': 50,
            'show-fields': 'headline,bodyText,byline,thumbnail',
            'api-key': api_key
        }
        
        articles = []
        
        try:
            # Get first page to see total pages
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            total_pages = data.get('response', {}).get('pages', 1)
            total_pages = min(total_pages, 20)  # Limit to 20 pages for rate limiting
            
            print(f"Guardian Historical: Fetching {total_pages} pages...")
            
            for page in range(1, total_pages + 1):
                params['page'] = page
                
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                for item in data.get('response', {}).get('results', []):
                    fields = item.get('fields', {})
                    articles.append({
                        'article_id': item.get('id', ''),
                        'title': fields.get('headline', item.get('webTitle', '')),
                        'description': fields.get('bodyText', '')[:500],
                        'content': fields.get('bodyText', ''),
                        'source': 'The Guardian',
                        'author': fields.get('byline', ''),
                        'url': item.get('webUrl', ''),
                        'published_at': item.get('webPublicationDate', ''),
                        'image_url': fields.get('thumbnail', ''),
                        'category': item.get('sectionName', 'general')
                    })
                
                print(f"Page {page}/{total_pages}: {len(articles)} total articles", end='\r')
                time.sleep(1)  # Rate limiting
            
            print(f"\n✓ Guardian: {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"✗ Guardian historical error: {e}")
            return []
    
    # ============ NY Times Archive API (ALL articles by month) ============
    def fetch_nytimes_archive_month(self, year, month, api_key):
        """
        Fetch ALL articles from NY Times for a given month using Archive API
        Goes back to 1851! Returns all articles (~20MB per month)
        
        Args:
            year: Year (e.g., 2020)
            month: Month (1-12)
            api_key: NY Times API key
        """
        url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
        
        params = {'api-key': api_key}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for doc in data.get('response', {}).get('docs', []):
                articles.append({
                    'article_id': doc.get('_id', ''),
                    'title': doc.get('headline', {}).get('main', ''),
                    'description': doc.get('abstract', ''),
                    'content': doc.get('lead_paragraph', ''),
                    'source': 'New York Times',
                    'author': doc.get('byline', {}).get('original', ''),
                    'url': doc.get('web_url', ''),
                    'published_at': doc.get('pub_date', ''),
                    'image_url': '',
                    'category': doc.get('section_name', 'general'),
                    'keywords': [kw.get('value', '') for kw in doc.get('keywords', [])]
                })
            
            return articles
            
        except Exception as e:
            print(f"✗ NY Times Archive error ({year}/{month}): {e}")
            return []
    
    def fetch_nytimes_archive_range(self, start_date_str, end_date_str, api_key):
        """
        Fetch ALL NY Times articles for a date range using Archive API
        
        Args:
            start_date_str: 'YYYY-MM-DD'
            end_date_str: 'YYYY-MM-DD'
            api_key: NY Times API key
        """
        from datetime import datetime
        from calendar import monthrange
        
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        all_articles = []
        
        print(f"NY Times Archive: Fetching from {start_date_str} to {end_date_str}")
        
        # Generate list of year/month pairs
        current_year = start_date.year
        current_month = start_date.month
        end_year = end_date.year
        end_month = end_date.month
        
        month_count = 0
        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            month_count += 1
            print(f"Month {month_count}: {current_year}-{current_month:02d}", end=" ")
            
            articles = self.fetch_nytimes_archive_month(current_year, current_month, api_key)
            all_articles.extend(articles)
            
            print(f"→ {len(articles)} articles (Total: {len(all_articles)})")
            
            # Rate limit: 10 requests per minute
            time.sleep(6)
            
            # Move to next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        
        print(f"\n✓ NY Times Archive: {len(all_articles)} total articles")
        return all_articles
    
    # ============ NY Times Historical Search (Legacy) ============
    def fetch_nytimes_historical(self, query, start_date_str, end_date_str, api_key):
        """
        Fetch historical data from NY Times using Search API (legacy method)
        Note: Archive API is better for comprehensive historical data
        
        Args:
            query: Search query
            start_date_str: 'YYYY-MM-DD'
            end_date_str: 'YYYY-MM-DD'
            api_key: NY Times API key
        """
        url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        
        begin_date = start_date_str.replace('-', '')
        end_date = end_date_str.replace('-', '')
        
        articles = []
        max_pages = 10  # NY Times limitsPages to 100, we'll do 10 for efficiency
        
        print(f"NY Times Historical: Fetching {max_pages} pages...")
        
        for page in range(max_pages):
            params = {
                'q': query,
                'begin_date': begin_date,
                'end_date': end_date,
                'sort': 'oldest',
                'page': page,
                'api-key': api_key
            }
            
            try:
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                docs = data.get('response', {}).get('docs', [])
                if not docs:
                    break
                
                for doc in docs:
                    articles.append({
                        'article_id': doc.get('_id', ''),
                        'title': doc.get('headline', {}).get('main', ''),
                        'description': doc.get('abstract', ''),
                        'content': doc.get('lead_paragraph', ''),
                        'source': 'New York Times',
                        'author': doc.get('byline', {}).get('original', ''),
                        'url': doc.get('web_url', ''),
                        'published_at': doc.get('pub_date', ''),
                        'image_url': '',
                        'category': doc.get('section_name', 'general')
                    })
                
                print(f"Page {page + 1}/{max_pages}: {len(articles)} total articles", end='\r')
                time.sleep(6)  # NY Times rate limit: 10 requests/minute
                
            except Exception as e:
                print(f"\n✗ NY Times error page {page}: {e}")
                break
        
        print(f"\n✓ NY Times: {len(articles)} articles")
        return articles
    
    # ============ Main Historical Fetch ============
    def fetch_historical_news(self, query, years_back=10, config_file='news_api_config.json'):
        """
        Main function to fetch historical news from all sources
        
        Args:
            query: Search query
            years_back: How many years of history to fetch
            config_file: Config file with API keys
        """
        # Load config
        config = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print("\n" + "="*70)
        print(f"FETCHING HISTORICAL NEWS: {years_back} YEARS")
        print(f"Date Range: {start_str} to {end_str}")
        print(f"Query: {query}")
        print("="*70 + "\n")
        
        all_articles = []
        
        # GDELT (Free, most comprehensive)
        print("\n[1/3] GDELT Project (1979-present)...")
        gdelt_articles = self.fetch_gdelt_timeline(query, start_str, end_str)
        all_articles.extend(gdelt_articles)
        
        # Guardian (if API key available)
        if config.get('guardian_key'):
            print("\n[2/3] The Guardian (1999-present)...")
            guardian_articles = self.fetch_guardian_historical(
                query, start_str, end_str, config['guardian_key']
            )
            all_articles.extend(guardian_articles)
        else:
            print("\n[2/3] Guardian: Skipped (no API key)")
        
        # NY Times (if API key available) - Use Archive API for comprehensive data
        if config.get('nytimes_key'):
            print("\n[3/3] New York Times Archive (1851-present)...")
            print("Using Archive API for comprehensive historical data...")
            nyt_articles = self.fetch_nytimes_archive_range(
                start_str, end_str, config['nytimes_key']
            )
            all_articles.extend(nyt_articles)
        else:
            print("\n[3/3] NY Times: Skipped (no API key)")
        
        print("\n" + "="*70)
        print(f"HISTORICAL FETCH COMPLETE: {len(all_articles)} TOTAL ARTICLES")
        print("="*70 + "\n")
        
        return all_articles

# Example usage
if __name__ == "__main__":
    fetcher = HistoricalNewsFetcher()
    
    # Fetch 10 years of historical data
    articles = fetcher.fetch_historical_news(
        query="economy OR finance OR stock market",
        years_back=10
    )
    
    print(f"\nFetched {len(articles)} historical articles")
    
    # Save to JSON
    with open('historical_news_data.json', 'w') as f:
        json.dump(articles, f, indent=2)
    print("Saved to historical_news_data.json")
