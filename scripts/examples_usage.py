"""
Example: Using the Tweet Predictor with OpenAI or Anthropic API

This shows how to integrate the prediction system into your own code
for higher quality AI-generated predictions.
"""

# ============================================================================
# EXAMPLE 1: Using OpenAI GPT-4 (Best Quality)
# ============================================================================

def example_openai():
    from fetch_news import get_current_news, format_news_for_context
    from predict_tweets import TweetPredictor
    
    # Get your API key from: https://platform.openai.com/
    OPENAI_API_KEY = "sk-..."  # Replace with your actual key
    
    # Optional: Get NewsAPI key from https://newsapi.org/ for real news
    NEWS_API_KEY = None  # Or use your real NewsAPI key
    
    print("Fetching current news...")
    news = get_current_news(api_key=NEWS_API_KEY, use_rss=True, max_articles=15)
    news_context = format_news_for_context(news, max_articles=10)
    
    print("Generating predictions with GPT-4...")
    predictor = TweetPredictor(model_type='openai')
    
    predictions = predictor.predict(
        news_context=news_context,
        api_key=OPENAI_API_KEY,
        topic_focus="economy and stock market",  # Optional focus
        num_tweets=5
    )
    
    # Display results
    for i, pred in enumerate(predictions, 1):
        print(f"\n=== TWEET {i} ===")
        print(f"Topic: {pred['topic']}")
        print(f"Related to: {pred['news_reference']}")
        print(f"\n{pred['tweet']}\n")
    
    # Save to file
    predictor.save_predictions(predictions)
    
    return predictions

# ============================================================================
# EXAMPLE 2: Using Anthropic Claude
# ============================================================================

def example_anthropic():
    from fetch_news import get_current_news, format_news_for_context
    from predict_tweets import TweetPredictor
    
    # Get your API key from: https://console.anthropic.com/
    ANTHROPIC_API_KEY = "sk-ant-..."  # Replace with your actual key
    
    print("Fetching current news...")
    news = get_current_news(use_rss=True)
    news_context = format_news_for_context(news)
    
    print("Generating predictions with Claude...")
    predictor = TweetPredictor(model_type='anthropic')
    
    predictions = predictor.predict(
        news_context=news_context,
        api_key=ANTHROPIC_API_KEY,
        num_tweets=3
    )
    
    for pred in predictions:
        print(f"\n{pred['tweet']}\n")
    
    return predictions

# ============================================================================
# EXAMPLE 3: Batch Processing Multiple Topics
# ============================================================================

def example_batch_topics():
    from fetch_news import get_current_news, format_news_for_context
    from predict_tweets import TweetPredictor
    
    OPENAI_API_KEY = "sk-..."  # Your API key
    
    # Get news once
    news = get_current_news(use_rss=True)
    news_context = format_news_for_context(news)
    
    predictor = TweetPredictor(model_type='openai')
    
    # Generate predictions for different topics
    topics = [
        "economy and jobs",
        "border security",
        "election integrity",
        "foreign policy"
    ]
    
    all_predictions = {}
    
    for topic in topics:
        print(f"\nGenerating tweets about: {topic}")
        
        predictions = predictor.predict(
            news_context=news_context,
            api_key=OPENAI_API_KEY,
            topic_focus=topic,
            num_tweets=2
        )
        
        all_predictions[topic] = predictions
        
        # Display
        for pred in predictions:
            print(f"  - {pred['tweet'][:100]}...")
    
    return all_predictions

# ============================================================================
# EXAMPLE 4: Using Custom News
# ============================================================================

def example_custom_news():
    from predict_tweets import TweetPredictor
    
    OPENAI_API_KEY = "sk-..."  # Your API key
    
    # Create your own custom news context
    custom_news = """
    CURRENT NEWS TOPICS:
    
    1. Federal Reserve Announces Interest Rate Decision
       The Fed maintains current rates amid strong economic indicators.
       Source: Financial Times
    
    2. New Infrastructure Bill Passes Senate
       Bipartisan infrastructure legislation approved with 65-35 vote.
       Source: Washington Post
    
    3. Major Tech Companies Report Record Earnings
       Apple, Microsoft, and Google exceed analyst expectations.
       Source: CNBC
    """
    
    predictor = TweetPredictor(model_type='openai')
    
    predictions = predictor.predict(
        news_context=custom_news,
        api_key=OPENAI_API_KEY,
        num_tweets=3
    )
    
    for pred in predictions:
        print(f"\n{pred['tweet']}\n")
    
    return predictions

# ============================================================================
# EXAMPLE 5: Save and Load Predictions
# ============================================================================

def example_save_load():
    import json
    from predict_tweets import TweetPredictor
    from fetch_news import get_current_news, format_news_for_context
    
    # Generate predictions
    news = get_current_news()
    news_context = format_news_for_context(news)
    
    predictor = TweetPredictor(model_type='local')  # Using template mode
    predictions = predictor.predict(news_context=news_context, num_tweets=5)
    
    # Save to custom filename
    predictor.save_predictions(predictions, filename='my_predictions.json')
    
    # Load predictions later
    with open('my_predictions.json', 'r') as f:
        loaded_data = json.load(f)
    
    print(f"Loaded {len(loaded_data['predictions'])} predictions")
    print(f"Generated at: {loaded_data['timestamp']}")
    print(f"Model used: {loaded_data['model_type']}")
    
    return loaded_data

# ============================================================================
# EXAMPLE 6: Error Handling
# ============================================================================

def example_error_handling():
    from fetch_news import get_current_news, format_news_for_context
    from predict_tweets import TweetPredictor
    
    try:
        news = get_current_news(use_rss=True)
        news_context = format_news_for_context(news)
        
        predictor = TweetPredictor(model_type='openai')
        
        # This might fail if API key is invalid
        predictions = predictor.predict(
            news_context=news_context,
            api_key="invalid-key",
            num_tweets=3
        )
        
        # Check for errors
        if isinstance(predictions, dict) and 'error' in predictions:
            print(f"Error occurred: {predictions['error']}")
            
            # Fall back to template-based
            print("Falling back to template-based generation...")
            predictor_fallback = TweetPredictor(model_type='local')
            predictions = predictor_fallback.predict(
                news_context=news_context,
                num_tweets=3
            )
        
        # Process predictions
        for pred in predictions:
            print(pred['tweet'])
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# ============================================================================
# MAIN: Choose which example to run
# ============================================================================

if __name__ == "__main__":
    print("Tweet Prediction Examples")
    print("=" * 60)
    print("\nAvailable examples:")
    print("1. OpenAI GPT-4 (requires API key)")
    print("2. Anthropic Claude (requires API key)")
    print("3. Batch process multiple topics")
    print("4. Custom news input")
    print("5. Save and load predictions")
    print("6. Error handling and fallbacks")
    print("\nNote: Replace API keys in the code before running!")
    print("\nTo run an example, uncomment the function call below:")
    print()
    print("# example_openai()")
    print("# example_anthropic()")
    print("# example_batch_topics()")
    print("# example_custom_news()")
    print("# example_save_load()")
    print("# example_error_handling()")
    
    # Uncomment to run:
    # example_save_load()  # This one works without API keys
