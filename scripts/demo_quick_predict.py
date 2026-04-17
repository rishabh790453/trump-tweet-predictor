"""
Quick Demo - Generate tweet predictions without needing API keys
This uses template-based generation for instant results
"""
from fetch_news import get_current_news, format_news_for_context
from predict_tweets import TweetPredictor
from datetime import datetime

print("=" * 80)
print("  TRUMP TWEET PREDICTOR - QUICK DEMO")
print("  Using template-based generation (no API key needed)")
print("=" * 80)

# Step 1: Get current news (using mock data)
print("\n[1/3] Fetching current news...")
news = get_current_news(api_key=None, use_rss=False, max_articles=10)
print(f"✓ Retrieved {len(news)} news articles")

print("\nTop headlines:")
for i, article in enumerate(news[:3], 1):
    print(f"  {i}. {article['title']}")

# Step 2: Format news context
news_context = format_news_for_context(news, max_articles=5)

# Step 3: Generate predictions
print("\n[2/3] Generating tweet predictions...")
predictor = TweetPredictor(model_type='local')  # Template-based, no API needed
predictions = predictor.predict(
    news_context=news_context,
    num_tweets=5
)

print(f"✓ Generated {len(predictions)} tweet predictions")

# Step 4: Display results
print("\n[3/3] PREDICTED TWEETS:")
print("=" * 80)

for i, pred in enumerate(predictions, 1):
    print(f"\n--- TWEET #{i} ---")
    print(pred['tweet'])
    print(f"\n[Length: {len(pred['tweet'])} chars | Topic: {pred['topic']}]")
    print("-" * 80)

# Step 5: Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = predictor.save_predictions(predictions, f"demo_predictions_{timestamp}.json")

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)
print(f"\nFor better quality predictions:")
print("  1. Get an API key from OpenAI (https://platform.openai.com/)")
print("  2. Run: python main_tweet_predictor.py")
print("  3. Select 'openai' and enter your API key")
print("\nOr use the system programmatically - see README_PREDICTOR.md")
print("=" * 80)
