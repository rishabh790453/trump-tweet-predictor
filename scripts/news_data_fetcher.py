"""
Comprehensive News Data Fetcher
Supports multiple major news sources with API integration
"""
import requests
from datetime import datetime, timedelta
import json
import time
import os
from typing import List, Dict, Optional
import hashlib

class NewsDataFetcher:
    """Fetch news from multiple sources"""
    
    def __init__(self, config_file='news_api_config.json'):
        """
        Initialize with API keys from config file
        
        Create news_api_config.json with:
        {
            "newsapi_key": "your_key",
            "guardian_key": "your_key",
            "nytimes_key": "your_key",
            "alphavantage_key": "your_key"
        }
        """
        self.config = self.load_config(config_file)
    
    def load_config(self, config_file):
        """Load API configuration"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: {config_file} not found. Using demo mode.")
            return {}
    
    def generate_article_id(self, url, title):
        """Generate unique article ID"""
        unique_str = f"{url}{title}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    # ============ NewsAPI.org ============
    def fetch_newsapi(self, query="economy OR politics OR finance", 
                      days_back=7, page_size=100):
        """
        Fetch from NewsAPI.org (free tier: 1 month history, 100 requests/day)
        Get key: https://newsapi.org/
        """
        api_key = self.config.get('newsapi_key')
        if not api_key:
            print("NewsAPI key not found. Skipping.")
            return []
        
        url = "https://newsapi.org/v2/everything"
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': page_size,
            'apiKey': api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'article_id': self.generate_article_id(
                        article.get('url', ''), 
                        article.get('title', '')
                    ),
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'source': article.get('source', {}).get('name', 'NewsAPI'),
                    'author': article.get('author', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'image_url': article.get('urlToImage', ''),
                    'category': 'general'
                })
            
            print(f"✓ NewsAPI: Fetched {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"✗ NewsAPI error: {e}")
            return []
    
    # ============ The Guardian ============
    def fetch_guardian(self, query="politics OR economy", days_back=30, page_size=50):
        """
        Fetch from The Guardian API (free, extensive historical archive)
        Get key: https://open-platform.theguardian.com/access/
        """
        api_key = self.config.get('guardian_key')
        if not api_key:
            print("Guardian API key not found. Skipping.")
            return []
        
        url = "https://content.guardianapis.com/search"
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from-date': from_date,
            'page-size': page_size,
            'show-fields': 'headline,bodyText,byline,thumbnail',
            'api-key': api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            articles = []
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
            
            print(f"✓ Guardian: Fetched {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"✗ Guardian error: {e}")
            return []
    
    # ============ New York Times ============
    def fetch_nytimes(self, query="economy", days_back=30):
        """
        Fetch from NY Times API (free tier: 4000 requests/day)
        Get key: https://developer.nytimes.com/
        """
        api_key = self.config.get('nytimes_key')
        if not api_key:
            print("NY Times API key not found. Skipping.")
            return []
        
        url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        
        begin_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
        end_date = datetime.now().strftime('%Y%m%d')
        
        params = {
            'q': query,
            'begin_date': begin_date,
            'end_date': end_date,
            'sort': 'newest',
            'api-key': api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
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
                    'category': doc.get('section_name', 'general')
                })
            
            print(f"✓ NY Times: Fetched {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"✗ NY Times error: {e}")
            return []
    
    # ============ Alpha Vantage News Sentiment ============
    def fetch_alphavantage_news(self, topics="economy,finance", limit=50):
        """
        Fetch from Alpha Vantage News Sentiment API (free)
        Get key: https://www.alphavantage.co/support/#api-key
        """
        api_key = self.config.get('alphavantage_key')
        if not api_key:
            print("Alpha Vantage API key not found. Skipping.")
            return []
        
        url = "https://www.alphavantage.co/query"
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'topics': topics,
            'limit': limit,
            'apikey': api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get('feed', []):
                articles.append({
                    'article_id': self.generate_article_id(item.get('url', ''), item.get('title', '')),
                    'title': item.get('title', ''),
                    'description': item.get('summary', ''),
                    'content': item.get('summary', ''),
                    'source': item.get('source', 'Alpha Vantage'),
                    'author': ', '.join([a.get('name', '') for a in item.get('authors', [])]),
                    'url': item.get('url', ''),
                    'published_at': item.get('time_published', ''),
                    'image_url': item.get('banner_image', ''),
                    'sentiment_score': float(item.get('overall_sentiment_score', 0)),
                    'category': 'finance'
                })
            
            print(f"✓ Alpha Vantage: Fetched {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"✗ Alpha Vantage error: {e}")
            return []
    
    # ============ RSS Feeds (Free, always available) ============
    def fetch_rss_feeds(self, custom_feeds=None):
        """
        Fetch from RSS feeds (free alternative, no API key required)
        """
        try:
            import feedparser
        except ImportError:
            print("feedparser not installed. Run: pip install feedparser")
            return []
        
        feeds = custom_feeds or [
            'http://rss.cnn.com/rss/cnn_topstories.rss',
            'http://feeds.bbci.co.uk/news/rss.xml',
            'https://feeds.npr.org/1001/rss.xml',
            'https://www.reuters.com/rssFeed/topNews',
            'https://www.politico.com/rss/politics08.xml',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'https://www.bloomberg.com/politics/feeds/site.xml',
        ]
        
        articles = []
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                source_name = feed.feed.get('title', 'RSS Feed')
                
                for entry in feed.entries[:20]:  # Top 20 per feed
                    published = entry.get('published', entry.get('updated', ''))
                    articles.append({
                        'article_id': self.generate_article_id(
                            entry.get('link', ''),
                            entry.get('title', '')
                        ),
                        'title': entry.get('title', ''),
                        'description': entry.get('summary', ''),
                        'content': entry.get('summary', ''),
                        'source': source_name,
                        'author': entry.get('author', ''),
                        'url': entry.get('link', ''),
                        'published_at': published,
                        'image_url': '',
                        'category': 'rss'
                    })
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"✗ RSS error ({feed_url}): {e}")
                continue
        
        print(f"✓ RSS Feeds: Fetched {len(articles)} articles")
        return articles
    
    def fetch_all(self, query="economy OR politics OR finance"):
        """Fetch from all available sources"""
        print("\n" + "="*60)
        print("FETCHING NEWS FROM ALL SOURCES")
        print("="*60 + "\n")
        
        all_articles = []
        
        # NewsAPI (recent news)
        all_articles.extend(self.fetch_newsapi(query=query, days_back=7))
        time.sleep(1)
        
        # Guardian (good historical coverage)
        all_articles.extend(self.fetch_guardian(query=query, days_back=30))
        time.sleep(1)
        
        # NY Times
        all_articles.extend(self.fetch_nytimes(query=query, days_back=30))
        time.sleep(1)
        
        # Alpha Vantage
        all_articles.extend(self.fetch_alphavantage_news(topics="economy,finance"))
        time.sleep(1)
        
        # RSS Feeds (always available)
        all_articles.extend(self.fetch_rss_feeds())
        
        print(f"\n{'='*60}")
        print(f"TOTAL ARTICLES FETCHED: {len(all_articles)}")
        print(f"{'='*60}\n")
        
        return all_articles

# Example usage
if __name__ == "__main__":
    fetcher = NewsDataFetcher()
    articles = fetcher.fetch_all()
    print(f"Fetched {len(articles)} total articles")
