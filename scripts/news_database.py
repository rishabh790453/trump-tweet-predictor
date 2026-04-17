"""
News Database Management
Handles storage and retrieval of news articles
"""
import sqlite3
import pandas as pd
from datetime import datetime
import json
import os

class NewsDatabase:
    def __init__(self, db_path='news_data.db'):
        """Initialize news database"""
        self.db_path = db_path
        self.create_tables()
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main news articles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id TEXT UNIQUE,
                title TEXT,
                description TEXT,
                content TEXT,
                source TEXT,
                author TEXT,
                url TEXT UNIQUE,
                published_at TIMESTAMP,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                category TEXT,
                sentiment_score REAL,
                keywords TEXT,
                image_url TEXT,
                lang TEXT DEFAULT 'en'
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_published_at ON news_articles(published_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON news_articles(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON news_articles(category)')
        
        # News sources tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT UNIQUE,
                source_type TEXT,
                api_endpoint TEXT,
                last_fetch TIMESTAMP,
                total_articles INTEGER DEFAULT 0,
                active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Fetch history table for monitoring
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fetch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT,
                fetch_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                articles_fetched INTEGER,
                status TEXT,
                error_message TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_article(self, article_data):
        """Insert a single article (skip if duplicate)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO news_articles 
                (article_id, title, description, content, source, author, url, 
                 published_at, category, sentiment_score, keywords, image_url, lang)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article_data.get('article_id', article_data.get('url', '')),
                article_data.get('title', ''),
                article_data.get('description', ''),
                article_data.get('content', ''),
                article_data.get('source', ''),
                article_data.get('author', ''),
                article_data.get('url', ''),
                article_data.get('published_at', ''),
                article_data.get('category', ''),
                article_data.get('sentiment_score'),
                json.dumps(article_data.get('keywords', [])),
                article_data.get('image_url', ''),
                article_data.get('lang', 'en')
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error inserting article: {e}")
            return False
        finally:
            conn.close()
    
    def insert_articles_batch(self, articles):
        """Insert multiple articles at once (more efficient)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted = 0
        for article in articles:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO news_articles 
                    (article_id, title, description, content, source, author, url, 
                     published_at, category, sentiment_score, keywords, image_url, lang)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.get('article_id', article.get('url', '')),
                    article.get('title', ''),
                    article.get('description', ''),
                    article.get('content', ''),
                    article.get('source', ''),
                    article.get('author', ''),
                    article.get('url', ''),
                    article.get('published_at', ''),
                    article.get('category', ''),
                    article.get('sentiment_score'),
                    json.dumps(article.get('keywords', [])) if article.get('keywords') else '[]',
                    article.get('image_url', ''),
                    article.get('lang', 'en')
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                print(f"Error inserting article: {e}")
                continue
        
        conn.commit()
        conn.close()
        return inserted
    
    def get_articles(self, start_date=None, end_date=None, source=None, limit=100):
        """Retrieve articles with filters"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM news_articles WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND published_at >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND published_at <= ?"
            params.append(end_date)
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        query += f" ORDER BY published_at DESC LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def get_stats(self):
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total articles
        cursor.execute("SELECT COUNT(*) FROM news_articles")
        total = cursor.fetchone()[0]
        
        # Articles by source
        cursor.execute("""
            SELECT source, COUNT(*) as count 
            FROM news_articles 
            GROUP BY source 
            ORDER BY count DESC
        """)
        by_source = cursor.fetchall()
        
        # Date range
        cursor.execute("SELECT MIN(published_at), MAX(published_at) FROM news_articles")
        date_range = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_articles': total,
            'by_source': by_source,
            'earliest_date': date_range[0],
            'latest_date': date_range[1]
        }
    
    def log_fetch(self, source_name, articles_count, status='success', error=None):
        """Log a fetch operation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO fetch_history (source_name, articles_fetched, status, error_message)
            VALUES (?, ?, ?, ?)
        ''', (source_name, articles_count, status, error))
        
        conn.commit()
        conn.close()
    
    def export_to_csv(self, output_file='news_export.csv', start_date=None, end_date=None):
        """Export articles to CSV"""
        df = self.get_articles(start_date=start_date, end_date=end_date, limit=1000000)
        df.to_csv(output_file, index=False)
        print(f"Exported {len(df)} articles to {output_file}")
        return output_file
