"""
N-gram Statistical Tweet Generator
Uses statistical word patterns from historic tweets
Learns what words commonly follow other words
"""
import pandas as pd
import re
import random
from collections import defaultdict, Counter
import json

class NgramTweetGenerator:
    def __init__(self, csv_file='trump_tweets_cleaned.csv', n=3):
        """
        Initialize N-gram generator
        
        Args:
            csv_file: Path to cleaned tweets CSV
            n: Size of n-grams (3 = trigram, 4 = 4-gram, etc.)
        """
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.unigrams = Counter()
        self.bigrams = defaultdict(Counter)
        self.sentence_starts = []
        
        print(f"Loading tweets from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} tweets")
        
        self.build_models()
    
    def tokenize(self, text):
        """Tokenize text into words and punctuation"""
        tokens = re.findall(r'\b[A-Z]{2,}\b|\b\w+\b|[!?.,:;]', str(text))
        return tokens
    
    def build_models(self):
        """Build n-gram models from all tweets"""
        print(f"Building {self.n}-gram model from tweets...")
        
        for content in self.df['content'].fillna(''):
            if not content or content == '[Image]' or content == '[Video]':
                continue
            
            tokens = self.tokenize(content)
            if len(tokens) < 2:
                continue
            
            # Track sentence starts
            if len(tokens) >= 2:
                self.sentence_starts.append((tokens[0], tokens[1]))
            
            # Build unigrams
            for token in tokens:
                self.unigrams[token] += 1
            
            # Build bigrams
            for i in range(len(tokens) - 1):
                self.bigrams[tokens[i]][tokens[i + 1]] += 1
            
            # Build n-grams
            for i in range(len(tokens) - self.n + 1):
                prefix = tuple(tokens[i:i + self.n - 1])
                next_word = tokens[i + self.n - 1]
                self.ngrams[prefix][next_word] += 1
        
        print(f"Built n-gram model:")
        print(f"  - {len(self.unigrams)} unique words")
        print(f"  - {len(self.bigrams)} bigram patterns")
        print(f"  - {len(self.ngrams)} {self.n}-gram patterns")
        print(f"  - {len(self.sentence_starts)} sentence starts")
    
    def get_next_word(self, prefix, temperature=1.0):
        """
        Get next word based on n-gram probabilities
        
        Args:
            prefix: Tuple of previous (n-1) words
            temperature: Randomness (higher = more random)
                        1.0 = normal, 0.5 = more predictable, 1.5 = more creative
        
        Returns:
            Next word
        """
        if prefix not in self.ngrams:
            # Fallback to bigram if n-gram not found
            if len(prefix) >= 1 and prefix[-1] in self.bigrams:
                choices = self.bigrams[prefix[-1]]
            else:
                # Random common word
                choices = self.unigrams
        else:
            choices = self.ngrams[prefix]
        
        if not choices:
            return None
        
        # Apply temperature to probabilities
        words = list(choices.keys())
        counts = list(choices.values())
        
        if temperature != 1.0:
            # Adjust probabilities with temperature
            total = sum(counts)
            probs = [c / total for c in counts]
            adjusted_probs = [p ** (1 / temperature) for p in probs]
            total_adjusted = sum(adjusted_probs)
            probs = [p / total_adjusted for p in adjusted_probs]
            return random.choices(words, weights=probs)[0]
        else:
            # Weighted random choice based on counts
            return random.choices(words, weights=counts)[0]
    
    def generate(self, max_length=500, start_with=None, temperature=1.0):
        """
        Generate a tweet using n-gram model
        
        Args:
            max_length: Maximum character length
            start_with: Optional starting word(s)
            temperature: Creativity (0.5-2.0, default 1.0)
        
        Returns:
            Generated tweet text
        """
        # Choose starting words
        if start_with:
            words = self.tokenize(start_with)
            if len(words) < self.n - 1:
                # Pad with a random start
                start = random.choice(self.sentence_starts)
                words = list(start) + words
        else:
            words = list(random.choice(self.sentence_starts))
        
        # Make sure we have enough words for prefix
        while len(words) < self.n - 1:
            words.insert(0, random.choice(list(self.unigrams.keys())))
        
        # Generate words
        attempts = 0
        max_attempts = 200
        
        while attempts < max_attempts:
            attempts += 1
            
            # Get prefix for n-gram lookup
            prefix = tuple(words[-(self.n - 1):])
            
            # Get next word
            next_word = self.get_next_word(prefix, temperature)
            if next_word is None:
                break
            
            words.append(next_word)
            
            # Check length
            tweet = self.reconstruct_text(words)
            if len(tweet) >= max_length:
                break
            
            # Stop at sentence endings if long enough
            if len(tweet) > 100 and next_word in ['.', '!', '?']:
                if random.random() < 0.6:  # 60% chance to stop
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
    
    def generate_with_keywords(self, keywords, num_tweets=3, max_length=500, temperature=1.0):
        """
        Generate tweets incorporating specific keywords
        
        Args:
            keywords: List of keywords to include
            num_tweets: Number of tweets to generate
            max_length: Max characters
            temperature: Creativity level
        
        Returns:
            List of generated tweets
        """
        tweets = []
        
        for _ in range(num_tweets):
            # Pick a random keyword to start with
            keyword = random.choice(keywords)
            tweet = self.generate(max_length=max_length, start_with=keyword, temperature=temperature)
            
            # Retry if no keywords present
            attempts = 0
            while attempts < 3 and not any(kw.lower() in tweet.lower() for kw in keywords):
                tweet = self.generate(max_length=max_length, temperature=temperature)
                attempts += 1
            
            tweets.append(tweet)
        
        return tweets
    
    def generate_from_news(self, news_articles, num_tweets=5, temperature=1.0):
        """
        Generate tweets based on news articles
        
        Args:
            news_articles: List of news article dicts
            num_tweets: Number of tweets to generate
            temperature: Creativity (0.5 = conservative, 1.5 = creative)
        
        Returns:
            List of tweet prediction dicts
        """
        predictions = []
        
        # Extract keywords from news
        all_keywords = []
        for article in news_articles[:10]:
            title = article.get('title', '')
            # Extract capitalized words (likely important nouns)
            keywords = re.findall(r'\b[A-Z][a-z]+\b', title)
            all_keywords.extend(keywords)
        
        # Mix with common Trump topics
        common_keywords = ['America', 'President', 'Great', 'Democrat', 
                          'Republican', 'Country', 'People', 'Stock', 
                          'Market', 'Border', 'Trade', 'Economy']
        
        keyword_pool = list(set(all_keywords + common_keywords))
        
        for i in range(num_tweets):
            # Select random keywords
            selected_keywords = random.sample(keyword_pool, min(3, len(keyword_pool)))
            
            # Generate tweet
            tweet = self.generate_with_keywords(
                selected_keywords, 
                num_tweets=1, 
                max_length=random.randint(150, 400),
                temperature=temperature
            )[0]
            
            predictions.append({
                'tweet': tweet,
                'topic': 'ngram-generated',
                'news_reference': news_articles[i % len(news_articles)].get('title', 'N/A') if news_articles else 'N/A',
                'method': f'{self.n}-gram Statistical Model',
                'keywords_used': selected_keywords,
                'temperature': temperature
            })
        
        return predictions
    
    def analyze_word_probability(self, prefix):
        """
        Show probability distribution for next word given prefix
        Useful for understanding the model
        """
        if isinstance(prefix, str):
            prefix = tuple(self.tokenize(prefix))
        
        if prefix not in self.ngrams:
            return f"No data for prefix: {prefix}"
        
        choices = self.ngrams[prefix]
        total = sum(choices.values())
        
        probs = [(word, count, count/total*100) for word, count in choices.most_common(10)]
        
        result = f"Top next words after '{' '.join(prefix)}':\n"
        for word, count, prob in probs:
            result += f"  {word}: {prob:.1f}% ({count} times)\n"
        
        return result

if __name__ == "__main__":
    print("=" * 80)
    print("  N-GRAM STATISTICAL TWEET GENERATOR")
    print("  Learning from 88K+ real tweets")
    print("=" * 80)
    
    # Build generator with trigrams
    generator = NgramTweetGenerator(n=3)
    
    print("\n" + "=" * 80)
    print("  SAMPLE GENERATED TWEETS (Temperature=1.0)")
    print("=" * 80)
    
    for i in range(5):
        print(f"\n--- TWEET {i+1} ---")
        tweet = generator.generate(max_length=300, temperature=1.0)
        print(tweet)
        print(f"\n[Length: {len(tweet)} chars]")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("  CREATIVE TWEETS (Temperature=1.5)")
    print("=" * 80)
    
    for i in range(3):
        print(f"\n--- CREATIVE TWEET {i+1} ---")
        tweet = generator.generate(max_length=250, temperature=1.5)
        print(tweet)
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("  KEYWORD-BASED GENERATION")
    print("=" * 80)
    
    keywords = ['Stock', 'Market', 'Record', 'Economy']
    keyword_tweets = generator.generate_with_keywords(keywords, num_tweets=2)
    
    for i, tweet in enumerate(keyword_tweets, 1):
        print(f"\n--- KEYWORD TWEET {i} ---")
        print(tweet)
        print("-" * 80)
    
    # Show probability analysis
    print("\n" + "=" * 80)
    print("  STATISTICAL ANALYSIS EXAMPLE")
    print("=" * 80)
    print("\n" + generator.analyze_word_probability(('The', 'Stock')))
    print(generator.analyze_word_probability(('MAKE', 'AMERICA')))
    
    print("\n✓ N-gram generator ready!")
    print("  Pure statistics from your real tweets - no API needed!")
