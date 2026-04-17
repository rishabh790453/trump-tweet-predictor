"""
Fetch current news from various sources for tweet prediction
"""
import requests
from datetime import datetime, timedelta
import json

def fetch_news_api(api_key=None, query="politics OR economy OR international", max_articles=20):
    """
    Fetch news from NewsAPI.org
    Get your API key from: https://newsapi.org/
    """
    if not api_key:
        print("Warning: No NewsAPI key provided. Using mock data.")
        return get_mock_news()
    
    url = "https://newsapi.org/v2/everything"
    
    # Get news from last 24 hours
    from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    params = {
        'q': query,
        'from': from_date,
        'sortBy': 'popularity',
        'language': 'en',
        'pageSize': max_articles,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        for article in data.get('articles', []):
            articles.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'source': article.get('source', {}).get('name', ''),
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', ''),
                'content': article.get('content', '')
            })
        
        return articles
    except Exception as e:
        print(f"Error fetching news from NewsAPI: {e}")
        return get_mock_news()

def fetch_rss_news(rss_urls=None):
    """
    Fetch news from RSS feeds (free alternative to NewsAPI)
    """
    try:
        import feedparser
    except ImportError:
        print("feedparser not installed. Run: pip install feedparser")
        return get_mock_news()
    
    if not rss_urls:
        rss_urls = [
            'http://rss.cnn.com/rss/cnn_topstories.rss',
            'http://feeds.bbci.co.uk/news/rss.xml',
            'https://www.politico.com/rss/politics08.xml',
            'https://www.reddit.com/r/worldnews/.rss',
        ]
    
    articles = []
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:  # Get top 5 from each feed
                articles.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', ''),
                    'source': feed.feed.get('title', 'RSS'),
                    'url': entry.get('link', ''),
                    'publishedAt': entry.get('published', ''),
                    'content': entry.get('summary', '')
                })
        except Exception as e:
            print(f"Error fetching RSS from {url}: {e}")
            continue
    
    return articles if articles else get_mock_news()

def get_mock_news():
    """
    Returns mock news data for testing when API is not available
    """
    return [
        {
            'title': 'Stock Market Reaches New Highs Amid Economic Growth',
            'description': 'Major stock indices hit record highs as economic indicators show strong growth and consumer confidence increases.',
            'source': 'Mock Financial News',
            'url': 'https://example.com/1',
            'publishedAt': datetime.now().isoformat(),
            'content': 'Stock markets continue their upward trajectory with the DOW reaching unprecedented levels...'
        },
        {
            'title': 'Border Security Measures Face Congressional Debate',
            'description': 'New border security legislation sparks heated debate in Congress over immigration policy and enforcement.',
            'source': 'Mock Political News',
            'url': 'https://example.com/2',
            'publishedAt': datetime.now().isoformat(),
            'content': 'Lawmakers are divided over proposed border security measures...'
        },
        {
            'title': 'International Trade Negotiations Continue',
            'description': 'Trade talks with multiple countries progress as officials work on new agreements.',
            'source': 'Mock International News',
            'url': 'https://example.com/3',
            'publishedAt': datetime.now().isoformat(),
            'content': 'International trade negotiations show signs of progress...'
        },
        {
            'title': 'Energy Independence Goals Show Progress',
            'description': 'New data shows significant progress toward national energy independence goals.',
            'source': 'Mock Energy News',
            'url': 'https://example.com/4',
            'publishedAt': datetime.now().isoformat(),
            'content': 'The nation moves closer to energy independence...'
        },
        {
            'title': 'Crime Rates Show Decline in Major Cities',
            'description': 'Latest statistics reveal decreasing crime rates across several major metropolitan areas.',
            'source': 'Mock Crime News',
            'url': 'https://example.com/5',
            'publishedAt': datetime.now().isoformat(),
            'content': 'Major cities report declining crime rates...'
        }
    ]

def get_current_news(api_key=None, use_rss=True, max_articles=20):
    """
    Get current news using available methods
    
    Args:
        api_key: NewsAPI key (optional)
        use_rss: Whether to try RSS feeds if API fails
        max_articles: Maximum number of articles to return
    
    Returns:
        List of news articles
    """
    articles = []
    
    # Try NewsAPI first if API key provided
    if api_key:
        articles = fetch_news_api(api_key, max_articles=max_articles)
    
    # If no articles yet and RSS is enabled, try RSS feeds
    if not articles and use_rss:
        articles = fetch_rss_news()
    
    # Fall back to mock data if nothing else worked
    if not articles:
        articles = get_mock_news()
    
    return articles[:max_articles]

def format_news_for_context(articles, max_articles=10):
    """
    Format news articles into a context string for AI models
    """
    context = "CURRENT NEWS TOPICS:\n\n"
    
    for i, article in enumerate(articles[:max_articles], 1):
        context += f"{i}. {article['title']}\n"
        if article.get('description'):
            context += f"   {article['description']}\n"
        context += f"   Source: {article['source']}\n\n"
    
    return context

if __name__ == "__main__":
    print("Fetching current news...\n")
    
    # You can provide your NewsAPI key here or leave as None to use mock/RSS data
    NEWS_API_KEY = None  # Replace with your API key from https://newsapi.org/
    
    news = get_current_news(api_key=NEWS_API_KEY, use_rss=True)
    
    print(f"Found {len(news)} articles\n")
    print("=" * 80)
    
    for i, article in enumerate(news[:5], 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Source: {article['source']}")
        print(f"   {article['description']}")
    
    print("\n" + "=" * 80)
    print("\nFormatted context for AI:")
    print(format_news_for_context(news, max_articles=5))
