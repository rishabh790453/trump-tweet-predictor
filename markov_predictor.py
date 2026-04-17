"""
Markov Chain Tweet Generator
Learns word sequences from historic tweets to generate new ones
No API needed - learns purely from your 88K tweets!
"""
import pandas as pd
import re
import random
from collections import defaultdict
import json

class MarkovTweetGenerator:
    def __init__(self, csv_file='trump_tweets_cleaned.csv', order=2):
        """
        Initialize Markov Chain generator
        
        Args:
            csv_file: Path to cleaned tweets CSV
            order: Chain order (2 = bigram, 3 = trigram)
                   Higher = more coherent but less creative
        """
        self.order = order
        self.chain = defaultdict(list)
        self.start_words = []
        
        print(f"Loading tweets from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} tweets")
        
        self.build_chain()
    
    def tokenize(self, text):
        """Split text into words while preserving punctuation"""
        # Keep words, punctuation, and ALL CAPS intact
        tokens = re.findall(r'\b[A-Z]{2,}\b|\b\w+\b|[!?.,:;]', str(text))
        return tokens
    
    def build_chain(self):
        """Build Markov chain from all tweets"""
        print("Building Markov chain from tweets...")
        
        tweet_count = 0
        for content in self.df['content'].fillna(''):
            if not content or content == '[Image]' or content == '[Video]':
                continue
            
            tokens = self.tokenize(content)
            if len(tokens) < self.order + 1:
                continue
            
            # Track starting words for each tweet
            start = tuple(tokens[:self.order])
            self.start_words.append(start)
            
            # Build the chain
            for i in range(len(tokens) - self.order):
                key = tuple(tokens[i:i + self.order])
                next_word = tokens[i + self.order]
                self.chain[key].append(next_word)
            
            tweet_count += 1
        
        print(f"Built chain from {tweet_count} tweets")
        print(f"Chain has {len(self.chain)} unique {self.order}-grams")
        print(f"Found {len(self.start_words)} possible starting sequences")
    
    def generate(self, max_length=500, start_with=None):
        """
        Generate a new tweet using the Markov chain
        
        Args:
            max_length: Maximum character length
            start_with: Optional word/phrase to start with
        
        Returns:
            Generated tweet text
        """
        if not self.chain:
            return "Error: Chain not built yet"
        
        # Choose starting words
        if start_with:
            # Try to find a sequence that matches the start
            start_with_lower = start_with.lower()
            matching_starts = [s for s in self.start_words 
                             if start_with_lower in ' '.join(s).lower()]
            if matching_starts:
                current = random.choice(matching_starts)
            else:
                current = random.choice(self.start_words)
        else:
            current = random.choice(self.start_words)
        
        words = list(current)
        
        # Generate until we hit a natural ending or max length
        attempts = 0
        max_attempts = 200
        
        while attempts < max_attempts:
            attempts += 1
            
            # Get next word
            if current not in self.chain:
                break
            
            next_word = random.choice(self.chain[current])
            words.append(next_word)
            
            # Update current state
            current = tuple(words[-self.order:])
            
            # Check if we should stop
            tweet = self.reconstruct_text(words)
            if len(tweet) >= max_length:
                break
            
            # Stop at sentence endings if tweet is long enough
            if len(tweet) > 100 and next_word in ['.', '!', '?']:
                if random.random() < 0.7:  # 70% chance to stop at sentence end
                    break
        
        return self.reconstruct_text(words)
    
    def reconstruct_text(self, words):
        """Reconstruct proper text from word list"""
        text = ''
        for i, word in enumerate(words):
            if word in ['.', '!', '?', ',', ':', ';']:
                text += word
            elif i == 0:
                text = word
            else:
                text += ' ' + word
        
        return text.strip()
    
    def generate_with_topic(self, topic_keywords, num_tweets=3, max_length=500):
        """
        Generate tweets that incorporate specific topics
        
        Args:
            topic_keywords: List of keywords to try to include
            num_tweets: Number of tweets to generate
            max_length: Max characters per tweet
        
        Returns:
            List of generated tweets
        """
        tweets = []
        
        for _ in range(num_tweets):
            # Try to start with topic-relevant words
            keyword = random.choice(topic_keywords)
            tweet = self.generate(max_length=max_length, start_with=keyword)
            
            # If keyword isn't in the tweet, try again
            attempts = 0
            while attempts < 5 and not any(kw.lower() in tweet.lower() for kw in topic_keywords):
                tweet = self.generate(max_length=max_length)
                attempts += 1
            
            tweets.append(tweet)
        
        return tweets
    
    def generate_from_news(self, news_articles, num_tweets=5):
        """
        Generate tweets based on news topics
        
        Args:
            news_articles: List of news article dicts with 'title' and 'description'
            num_tweets: Number of tweets to generate
        
        Returns:
            List of tweet dicts with metadata
        """
        predictions = []
        
        # Extract keywords from news
        all_topics = []
        for article in news_articles[:10]:
            title_words = re.findall(r'\b[A-Z][a-z]+\b', article.get('title', ''))
            all_topics.extend(title_words)
        
        # Common topics to mix in
        common_topics = ['Stock', 'Market', 'Border', 'Democrat', 'America', 
                        'President', 'Great', 'Country', 'Trade', 'Economy']
        
        for i in range(num_tweets):
            # Mix news topics with common topics
            topic_pool = all_topics + common_topics
            topic_keywords = random.sample(topic_pool, min(3, len(topic_pool)))
            
            tweet = self.generate_with_topic(topic_keywords, num_tweets=1, max_length=400)[0]
            
            # Format the prediction
            predictions.append({
                'tweet': tweet,
                'topic': 'markov-generated',
                'news_reference': news_articles[i % len(news_articles)].get('title', 'N/A') if news_articles else 'N/A',
                'method': 'Markov Chain',
                'keywords_used': topic_keywords
            })
        
        return predictions
    
    def save_chain(self, filename='markov_chain.json'):
        """Save the Markov chain to a file"""
        chain_dict = {str(k): v for k, v in self.chain.items()}
        start_words_list = [list(s) for s in self.start_words]
        
        data = {
            'order': self.order,
            'chain': chain_dict,
            'start_words': start_words_list
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        print(f"Markov chain saved to {filename}")
    
    def load_chain(self, filename='markov_chain.json'):
        """Load a saved Markov chain"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.order = data['order']
        self.chain = defaultdict(list)
        for k, v in data['chain'].items():
            self.chain[eval(k)] = v
        self.start_words = [tuple(s) for s in data['start_words']]
        
        print(f"Markov chain loaded from {filename}")

if __name__ == "__main__":
    print("=" * 80)
    print("  MARKOV CHAIN TWEET GENERATOR")
    print("  Learning from 88K+ real tweets")
    print("=" * 80)
    
    # Build the generator
    generator = MarkovTweetGenerator(order=2)
    
    print("\n" + "=" * 80)
    print("  SAMPLE GENERATED TWEETS")
    print("=" * 80)
    
    # Generate some sample tweets
    for i in range(5):
        print(f"\n--- TWEET {i+1} ---")
        tweet = generator.generate(max_length=300)
        print(tweet)
        print(f"\n[Length: {len(tweet)} chars]")
        print("-" * 80)
    
    # Try topic-based generation
    print("\n" + "=" * 80)
    print("  TOPIC-BASED GENERATION (Economy)")
    print("=" * 80)
    
    economy_tweets = generator.generate_with_topic(
        ['Stock', 'Market', 'Economy', 'Jobs', 'Trade'],
        num_tweets=3
    )
    
    for i, tweet in enumerate(economy_tweets, 1):
        print(f"\n--- ECONOMY TWEET {i} ---")
        print(tweet)
        print("-" * 80)
    
    print("\n✓ Markov chain generator ready!")
    print("  This uses ONLY your real tweets - no API needed!")
