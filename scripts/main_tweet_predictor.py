"""
Main Tweet Prediction System
Combines historic tweet analysis with current news to predict new tweets
"""
import os
from datetime import datetime
from fetch_news import get_current_news, format_news_for_context
from analyze_tweet_patterns import TweetPatternAnalyzer
from predict_tweets import TweetPredictor

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def main():
    print_header("TRUMP TWEET PREDICTION SYSTEM")
    
    # Configuration
    print("Configuration Options:")
    print("1. Model Type:")
    print("   - 'openai' - Uses GPT-4 (requires API key, best quality)")
    print("   - 'anthropic' - Uses Claude (requires API key, best quality)")
    print("   - 'markov' - Markov chains (FREE, learns from your tweets)")
    print("   - 'ngram' - N-gram statistics (FREE, learns from your tweets)")
    print("   - 'template' - Simple templates (FREE, basic quality)")
    print()
    
    model_type = input("Select model type (openai/anthropic/markov/ngram/template) [markov]: ").strip().lower()
    if model_type not in ['openai', 'anthropic', 'markov', 'ngram', 'template']:
        model_type = 'markov'  # Default to markov instead of template
    
    api_key = None
    if model_type in ['openai', 'anthropic']:
        api_key = input(f"Enter your {model_type.upper()} API key (or press Enter to skip): ").strip()
        if not api_key:
            print(f"No API key provided. Switching to Markov chain mode.")
            model_type = 'markov'
    
    temperature = 1.0
    if model_type == 'ngram':
        temp_input = input("Temperature for creativity (0.5=conservative, 1.0=normal, 1.5=creative) [1.0]: ").strip()
        try:
            temperature = float(temp_input) if temp_input else 1.0
        except:
            temperature = 1.0
    
    news_source = input("Use (1) NewsAPI, (2) RSS feeds, or (3) Mock data? [3]: ").strip()
    use_news_api = news_source == '1'
    use_rss = news_source == '2'
    
    news_api_key = None
    if use_news_api:
        news_api_key = input("Enter NewsAPI key (get from newsapi.org): ").strip()
    
    num_tweets = input("How many tweets to generate? [5]: ").strip()
    num_tweets = int(num_tweets) if num_tweets.isdigit() else 5
    
    # Step 1: Analyze Historic Tweets
    print_header("STEP 1: Analyzing Historic Tweet Patterns")
    
    if not os.path.exists('tweet_patterns.json'):
        print("Analyzing 88,000+ historic tweets...")
        analyzer = TweetPatternAnalyzer('trump_tweets_cleaned.csv')
        patterns = analyzer.analyze_all_patterns()
        print(analyzer.get_writing_style_summary())
        analyzer.save_patterns()
    else:
        print("Using existing tweet patterns from tweet_patterns.json")
        print("(Delete this file to re-analyze)")
    
    # Step 2: Fetch Current News
    print_header("STEP 2: Fetching Current News")
    
    print("Fetching latest news articles...")
    news = get_current_news(
        api_key=news_api_key if use_news_api else None,
        use_rss=use_rss,
        max_articles=15
    )
    
    print(f"Retrieved {len(news)} news articles")
    print("\nTop Headlines:")
    for i, article in enumerate(news[:5], 1):
        print(f"{i}. {article['title']}")
        print(f"   Source: {article['source']}")
    
    news_context = format_news_for_context(news, max_articles=10)
    
    # Add temperature parameter for ngram
    if model_type == 'ngram':
        predictions = predictor.predict(
            news_context=news_context,
            api_key=api_key,
            num_tweets=num_tweets,
            temperature=temperature
        )
    else:
        predictions = predictor.predict(
            news_context=news_context,
            api_key=api_key,
            num_tweets=num_tweets
        print(f"Using {model_type.upper()} model to generate {num_tweets} predictions...")
    print("This may take a moment...\n")
    
    predictor = TweetPredictor(model_type=model_type)
    predictions = predictor.predict(
        news_context=news_context,
        api_key=api_key,
        num_tweets=num_tweets
    )
    
    # Step 4: Display Results
    print_header("PREDICTED TWEETS")
    
    if isinstance(predictions, dict) and 'error' in predictions:
        print(f"Error: {predictions['error']}")
        if 'raw_response' in predictions:
            print("\nRaw Response:")
            print(predictions['raw_response'])
    elif isinstance(predictions, list):
        for i, pred in enumerate(predictions, 1):
            print(f"\n--- TWEET {i} ---")
            if isinstance(pred, dict):
                print(f"Topic: {pred.get('topic', 'N/A')}")
                print(f"Related to: {pred.get('news_reference', 'N/A')}")
                print(f"\nTweet:")
                print(pred.get('tweet', str(pred)))
                print(f"\nLength: {len(pred.get('tweet', ''))} characters")
            else:
                print(pred)
            print("-" * 80)
        
        # Save predictions
        save = input("\nSave predictions to file? (y/n) [y]: ").strip().lower()
        if save != 'n':
            filename = predictor.save_predictions(predictions)
            print(f"\n✓ Predictions saved to: {filename}")
    else:
        print("Unexpected response format:")
        print(predictions)
    
    # Step 5: Summary
    print_header("SUMMARY")
    print(f"Model used: {model_type.upper()}")
    print(f"News articles analyzed: {len(news)}")
    print(f"Tweets generated: {len(predictions) if isinstance(predictions, list) else 'Error'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "=" * 80)
    print("Tweet prediction complete!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrediction cancelled by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
