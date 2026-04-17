"""
Tweet Prediction Engine using AI/LLM or Statistical Models
Supports: OpenAI, Anthropic, Markov chains, N-grams, or templates
"""
import json
import random
from datetime import datetime

class TweetPredictor:
    def __init__(self, patterns_file='tweet_patterns.json', model_type='openai'):
        """
        Initialize the tweet predictor
        
        Args:
            patterns_file: Path to the analyzed patterns JSON file
            model_type: 'openai', 'anthropic', 'markov', 'ngram', or 'template'
        """
        self.model_type = model_type
        self.markov_generator = None
        self.ngram_generator = None
        
        # Load patterns if available
        try:
            with open(patterns_file, 'r') as f:
                self.patterns = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {patterns_file} not found. Run analyze_tweet_patterns.py first.")
            self.patterns = {}
    
    def create_system_prompt(self):
        """Create the system prompt with writing style instructions"""
        prompt = """You are tasked with predicting tweets in the style of Donald Trump based on current news.

WRITING STYLE CHARACTERISTICS:
1. Use ALL CAPS for emphasis on key words and phrases (TREMENDOUS, GREAT, FAKE NEWS, etc.)
2. Use lots of exclamation marks!
3. Use superlatives frequently: greatest, best, worst, biggest, most, tremendous, fantastic
4. Express strong opinions with confidence
5. Often criticize opponents with nicknames (Radical Left, Crooked, Sleepy, Do Nothing, etc.)
6. Praise supporters effusively with phrases like "Complete and Total Endorsement"
7. Reference achievements and records frequently
8. Use signature phrases:
   - "Thank you for your attention to this matter!"
   - "will never let you down"
   - "the likes of which nobody has ever seen"
   - "in the history of our Country"
   - "MAKE AMERICA GREAT AGAIN"
   - "America First"
   - "Law and Order"

9. Often sign tweets: "PRESIDENT DONALD J. TRUMP" or "President DJT"
10. Length: Typically 100-500 characters, but can be much longer for important announcements
11. Structure: Often multiple sentences, sometimes with dramatic build-up
12. Topics frequently mentioned: Economy, Stock Market, Border Security, Crime, Military, Elections, Media, Political Opponents

TONE: Confident, assertive, promotional, often combative with critics, praising of allies"""
        
        return prompt
    
    def create_prediction_prompt(self, news_context, topic_focus=None, num_tweets=3):
        """
        Create the user prompt for generating tweet predictions
        
        Args:
            news_context: String containing current news summaries
            topic_focus: Optional specific topic to focus on
            num_tweets: Number of tweets to generate
        """
        prompt = f"""Based on the following current news, predict {num_tweets} tweets that Donald Trump might post:

{news_context}

"""
        if topic_focus:
            prompt += f"Focus particularly on: {topic_focus}\n\n"
        
        prompt += f"""Generate {num_tweets} different tweets that:
1. React to or comment on the current news events
2. Match the writing style described in the system prompt
3. Are authentic to his communication patterns
4. Vary in topic and tone (some may be critical, some promotional, some informational)

Format your response as a JSON array of tweets, like this:
[
  {{
    "tweet": "The full tweet text here...",
    "topic": "brief topic description",
    "news_reference": "which news item this relates to"
  }}
]
"""
        return prompt
    
    def predict_with_openai(self, news_context, api_key, topic_focus=None, num_tweets=3):
        """Generate predictions using OpenAI API"""
        try:
            import openai
        except ImportError:
            return {"error": "openai package not installed. Run: pip install openai"}
        
        try:
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",  # or "gpt-3.5-turbo" for faster/cheaper
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": self.create_prediction_prompt(news_context, topic_focus, num_tweets)}
                ],
                temperature=0.8,  # Higher temperature for more creativity
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                # Extract JSON from markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                predictions = json.loads(content)
                return predictions
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw content
                return {"raw_response": content, "error": "Could not parse JSON"}
                
        except Exception as e:
            return {"error": f"OpenAI API error: {str(e)}"}
    
    def predict_with_anthropic(self, news_context, api_key, topic_focus=None, num_tweets=3):
        """Generate predictions using Anthropic Claude API"""
        try:
            import anthropic
        except ImportError:
            return {"error": "anthropic package not installed. Run: pip install anthropic"}
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            
            message = client.messages.create(
                model="claude-3-opus-20240229",  # or claude-3-sonnet for faster/cheaper
                max_tokens=2000,
                temperature=0.8,
                system=self.create_system_prompt(),
                messages=[
                    {"role": "user", "content": self.create_prediction_prompt(news_context, topic_focus, num_tweets)}
                ]
            )
            
            content = message.content[0].text
            
            # Try to parse JSON response
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                predictions = json.loads(content)
                return predictions
            except json.JSONDecodeError:
                return {"raw_response": content, "error": "Could not parse JSON"}
                
        except Exception as e:
            return {"markov(self, news_context, topic_focus=None, num_tweets=3):
        """Generate predictions using Markov Chain model"""
        try:
            from markov_predictor import MarkovTweetGenerator
        except ImportError:
            return {"error": "markov_predictor.py not found"}
        
        print("Using Markov Chain generator (learned from your 88K tweets)...")
        
        if self.markov_generator is None:
            self.markov_generator = MarkovTweetGenerator(order=2)
        
        # Extract news for context
        from fetch_news import get_current_news
        news = get_current_news(use_rss=False)  # Use mock data or pass actual news
        
        predictions = self.markov_generator.generate_from_news(news, num_tweets=num_tweets)
        return predictions
    
    def predict_with_ngram(self, news_context, topic_focus=None, num_tweets=3, temperature=1.0):
        """Generate predictions using N-gram statistical model"""
        try:
            from ngram_predictor import NgramTweetGenerator
        except ImportError:
            return {"error": "ngram_predictor.py not found"}
        
        print("Using N-gram statistical model (learned from your 88K tweets)...")
        
        if self.ngram_generator is None:
            self.ngram_generator = NgramTweetGenerator(n=3)
        
        # Extract news for context
        from fetch_news import get_current_news
        news = get_current_news(use_rss=False)
        
        predictions = self.ngram_generator.generate_from_news(
            news, 
            num_tweets=num_tweets,, temperature=1.0):
        """
        Main prediction method that routes to appropriate model
        
        Args:
            news_context: String containing current news
            api_key: API key for the chosen model (only needed for openai/anthropic)
            topic_focus: Optional topic to focus on
            num_tweets: Number of tweets to generate
            temperature: Creativity for ngram model (0.5-2.0)
        
        Returns:
            List of predicted tweets with metadata
        """
        if self.model_type == 'openai' and api_key:
            return self.predict_with_openai(news_context, api_key, topic_focus, num_tweets)
        elif self.model_type == 'anthropic' and api_key:
            return self.predict_with_anthropic(news_context, api_key, topic_focus, num_tweets)
        elif self.model_type == 'markov':
            return self.predict_with_markov(news_context, topic_focus, num_tweets)
        elif self.model_type == 'ngram':
            return self.predict_with_ngram(news_context, topic_focus, num_tweets, temperature)
        elif self.model_type == 'template':
            return self.predict_with_local_model(news_context, topic_focus, num_tweets)
        else:
            # Default fallback
        adjectives = ["TREMENDOUS", "TERRIBLE", "FANTASTIC", "DISASTROUS", "INCREDIBLE", "HORRIBLE", "AMAZING"]
        topics = ["the Economy", "Border Security", "Crime", "the Stock Market", "International Trade"]
        solutions = ["STRONG LEADERSHIP", "LAW AND ORDER", "America First policies", "REAL change"]
        achievements = ["RECORD economic growth", "the STRONGEST military", "HISTORIC trade deals"]
        signatures = ["MAKE AMERICA GREAT AGAIN!", "President DJT", "Thank you for your attention to this matter! PRESIDENT DONALD J. TRUMP"]
        
        predictions = []
        for i in range(num_tweets):
            template = random.choice(templates)
            tweet = template.format(
                topic=random.choice(topics),
                adjective=random.choice(adjectives),
                achievement=random.choice(achievements),
                solution=random.choice(solutions),
                signature=random.choice(signatures)
            )
            predictions.append({
                "tweet": tweet,
                "topic": "template-generated",
                "news_reference": "N/A (template-based)"
            })
        
        return predictions
    
    def predict(self, news_context, api_key=None, topic_focus=None, num_tweets=3):
        """
        Main prediction method that routes to appropriate model
        
        Args:
            news_context: String containing current news
            api_key: API key for the chosen model
            topic_focus: Optional topic to focus on
            num_tweets: Number of tweets to generate
        
        Returns:
            List of predicted tweets with metadata
        """
        if self.model_type == 'openai' and api_key:
            return self.predict_with_openai(news_context, api_key, topic_focus, num_tweets)
        elif self.model_type == 'anthropic' and api_key:
            return self.predict_with_anthropic(news_context, api_key, topic_focus, num_tweets)
        else:
            return self.predict_with_local_model(news_context, topic_focus, num_tweets)
    
    def save_predictions(self, predictions, filename=None):
        """Save predictions to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predicted_tweets_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model_type': self.model_type,
                'predictions': predictions
            }, f, indent=2)
        
        print(f"Predictions saved to {filename}")
        return filename

if __name__ == "__main__":
    print("Tweet Prediction Engine")
    print("=" * 60)
    print("\nThis module requires:")
    print("1. Run analyze_tweet_patterns.py first to generate patterns")
    print("2. API key from OpenAI or Anthropic (or use template mode)")
    print("\nExample usage:")
    print("""
from predict_tweets import TweetPredictor
from fetch_news import get_current_news, format_news_for_context

# Get current news
news = get_current_news()
news_context = format_news_for_context(news)

# Create predictor
predictor = TweetPredictor(model_type='openai')  # or 'anthropic' or 'local'

# Generate predictions
predictions = predictor.predict(
    news_context=news_context,
    api_key='your-api-key-here',  # Not needed for 'local' model
    num_tweets=5
)

# Save predictions
predictor.save_predictions(predictions)
""")
