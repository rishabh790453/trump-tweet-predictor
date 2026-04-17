"""
Analyze patterns in historic Trump tweets
"""
import pandas as pd
import re
from collections import Counter, defaultdict
import json

class TweetPatternAnalyzer:
    def __init__(self, csv_file='trump_tweets_cleaned.csv'):
        self.df = pd.read_csv(csv_file)
        self.patterns = {}
    
    def analyze_all_patterns(self):
        """Run all pattern analyses"""
        print("Analyzing tweet patterns...\n")
        
        self.patterns['common_phrases'] = self.extract_common_phrases()
        self.patterns['capitalization'] = self.analyze_capitalization()
        self.patterns['punctuation'] = self.analyze_punctuation()
        self.patterns['topics'] = self.extract_topics()
        self.patterns['sentence_structure'] = self.analyze_sentence_structure()
        self.patterns['common_targets'] = self.extract_common_targets()
        self.patterns['signature_phrases'] = self.extract_signature_phrases()
        self.patterns['length_stats'] = self.analyze_length_stats()
        
        return self.patterns
    
    def extract_common_phrases(self, top_n=50):
        """Extract most common multi-word phrases"""
        all_text = ' '.join(self.df['content'].fillna('').astype(str))
        
        # Extract 2-3 word phrases
        bigrams = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', all_text)
        trigrams = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b', all_text)
        
        common_bigrams = Counter(bigrams).most_common(top_n)
        common_trigrams = Counter(trigrams).most_common(top_n)
        
        return {
            'bigrams': common_bigrams,
            'trigrams': common_trigrams
        }
    
    def analyze_capitalization(self):
        """Analyze use of ALL CAPS for emphasis"""
        all_caps_words = []
        
        for content in self.df['content'].fillna(''):
            # Find words that are all caps (at least 3 letters)
            caps_words = re.findall(r'\b([A-Z]{3,})\b', str(content))
            all_caps_words.extend(caps_words)
        
        common_caps = Counter(all_caps_words).most_common(100)
        
        # Calculate percentage of tweets with caps
        tweets_with_caps = sum(1 for content in self.df['content'].fillna('') 
                               if re.search(r'\b[A-Z]{3,}\b', str(content)))
        caps_percentage = (tweets_with_caps / len(self.df)) * 100
        
        return {
            'common_caps_words': common_caps,
            'percentage_tweets_with_caps': caps_percentage,
            'avg_caps_per_tweet': len(all_caps_words) / len(self.df)
        }
    
    def analyze_punctuation(self):
        """Analyze punctuation patterns"""
        exclamations = 0
        questions = 0
        quotes = 0
        
        for content in self.df['content'].fillna(''):
            content_str = str(content)
            exclamations += content_str.count('!')
            questions += content_str.count('?')
            quotes += content_str.count('"')
        
        return {
            'avg_exclamations_per_tweet': exclamations / len(self.df),
            'avg_questions_per_tweet': questions / len(self.df),
            'avg_quotes_per_tweet': quotes / len(self.df),
            'tweets_with_exclamation_pct': sum(1 for c in self.df['content'].fillna('') 
                                                if '!' in str(c)) / len(self.df) * 100
        }
    
    def extract_topics(self):
        """Extract common topics and keywords"""
        keywords = {
            'democrats': ['democrat', 'democrats', 'schumer', 'pelosi', 'biden', 'radical left'],
            'economy': ['economy', 'jobs', 'gdp', 'stock market', 'dow', 'inflation', 'trade'],
            'border': ['border', 'immigration', 'wall', 'illegal'],
            'crime': ['crime', 'law and order', 'law enforcement', 'police'],
            'military': ['military', 'armed forces', 'defense', 'veterans'],
            'election': ['election', 'vote', 'voter', 'ballot', 'fraud'],
            'media': ['fake news', 'media', 'cnn', 'nbc', 'new york times'],
            'international': ['china', 'russia', 'north korea', 'iran', 'trade'],
            'achievements': ['tremendous', 'record', 'best ever', 'historic', 'greatest']
        }
        
        topic_counts = defaultdict(int)
        
        for content in self.df['content'].fillna(''):
            content_lower = str(content).lower()
            for topic, words in keywords.items():
                if any(word in content_lower for word in words):
                    topic_counts[topic] += 1
        
        total = len(self.df)
        topic_percentages = {topic: (count / total * 100) 
                           for topic, count in topic_counts.items()}
        
        return dict(sorted(topic_percentages.items(), 
                          key=lambda x: x[1], reverse=True))
    
    def analyze_sentence_structure(self):
        """Analyze sentence structure patterns"""
        sentence_counts = []
        
        for content in self.df['content'].fillna(''):
            # Count sentences (approximate)
            sentences = re.split(r'[.!?]+', str(content))
            sentence_counts.append(len([s for s in sentences if s.strip()]))
        
        return {
            'avg_sentences_per_tweet': sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0,
            'median_sentences': sorted(sentence_counts)[len(sentence_counts)//2] if sentence_counts else 0
        }
    
    def extract_common_targets(self):
        """Extract commonly mentioned people/entities (often criticized)"""
        # Common patterns for mentioning people/entities
        mentions = []
        
        patterns = [
            r'Radical (\w+)',
            r'Crooked (\w+)',
            r'Sleepy (\w+)',
            r'Crazy (\w+)',
            r'Corrupt (\w+)',
            r'Do Nothing (\w+)',
            r'Fake (\w+)',
        ]
        
        for content in self.df['content'].fillna(''):
            for pattern in patterns:
                matches = re.findall(pattern, str(content))
                mentions.extend(matches)
        
        return Counter(mentions).most_common(30)
    
    def extract_signature_phrases(self):
        """Extract Trump's signature phrases"""
        signature_phrases = [
            'MAKE AMERICA GREAT AGAIN',
            'Thank you for your attention to this matter',
            'PRESIDENT DONALD J. TRUMP',
            'Complete and Total',
            'will never let you down',
            'has my complete and total endorsement',
            'Fake News',
            'Witch Hunt',
            'Trump Derangement Syndrome',
            'Law and Order',
            'America First',
            'the likes of which',
            'nobody has ever seen',
            'in the history of',
            'the Greatest',
            'Tremendous',
        ]
        
        phrase_counts = {}
        for phrase in signature_phrases:
            count = sum(1 for content in self.df['content'].fillna('') 
                       if phrase.lower() in str(content).lower())
            if count > 0:
                phrase_counts[phrase] = count
        
        return dict(sorted(phrase_counts.items(), 
                          key=lambda x: x[1], reverse=True))
    
    def analyze_length_stats(self):
        """Analyze tweet length statistics"""
        lengths = [len(str(content)) for content in self.df['content'].fillna('')]
        
        return {
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'median_length': sorted(lengths)[len(lengths)//2] if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
        }
    
    def get_writing_style_summary(self):
        """Generate a comprehensive writing style summary"""
        if not self.patterns:
            self.analyze_all_patterns()
        
        summary = f"""
TRUMP TWEET WRITING STYLE ANALYSIS
=====================================

LENGTH:
- Average: {self.patterns['length_stats']['avg_length']:.0f} characters
- Median: {self.patterns['length_stats']['median_length']:.0f} characters
- Range: {self.patterns['length_stats']['min_length']:.0f} - {self.patterns['length_stats']['max_length']:.0f} characters

CAPITALIZATION:
- {self.patterns['capitalization']['percentage_tweets_with_caps']:.1f}% of tweets use ALL CAPS
- Average {self.patterns['capitalization']['avg_caps_per_tweet']:.1f} caps words per tweet
- Top ALL CAPS words: {', '.join([w[0] for w in self.patterns['capitalization']['common_caps_words'][:10]])}

PUNCTUATION:
- Average {self.patterns['punctuation']['avg_exclamations_per_tweet']:.1f} exclamation marks per tweet
- {self.patterns['punctuation']['tweets_with_exclamation_pct']:.1f}% of tweets contain exclamation marks

COMMON TOPICS (% of tweets):
{chr(10).join([f'- {topic}: {pct:.1f}%' for topic, pct in list(self.patterns['topics'].items())[:10]])}

TOP SIGNATURE PHRASES:
{chr(10).join([f'- "{phrase}": {count} times' for phrase, count in list(self.patterns['signature_phrases'].items())[:10]])}

SENTENCE STRUCTURE:
- Average {self.patterns['sentence_structure']['avg_sentences_per_tweet']:.1f} sentences per tweet
"""
        return summary
    
    def save_patterns(self, filename='tweet_patterns.json'):
        """Save patterns to JSON file"""
        if not self.patterns:
            self.analyze_all_patterns()
        
        # Convert to JSON-serializable format
        json_patterns = {}
        for key, value in self.patterns.items():
            if isinstance(value, dict):
                json_patterns[key] = value
            else:
                json_patterns[key] = str(value)
        
        with open(filename, 'w') as f:
            json.dump(json_patterns, f, indent=2)
        
        print(f"Patterns saved to {filename}")

if __name__ == "__main__":
    analyzer = TweetPatternAnalyzer()
    patterns = analyzer.analyze_all_patterns()
    
    print(analyzer.get_writing_style_summary())
    
    analyzer.save_patterns()
