# Trump Tweet Predictor

An AI-powered system that predicts Donald Trump's tweets based on historic tweet patterns and current news.

## Overview

This system combines:
- **Historic Tweet Analysis**: Analyzes 88,000+ historic tweets to learn writing patterns, style, and topics
- **Current News Fetching**: Retrieves latest news from NewsAPI, RSS feeds, or uses mock data
- **AI-Powered Prediction**: Uses GPT-4, Claude, or template-based generation to predict tweets

## Files

- `main_tweet_predictor.py` - Main interactive script
- `analyze_tweet_patterns.py` - Analyzes historic tweets for patterns
- `fetch_news.py` - Fetches current news from various sources
- `predict_tweets.py` - AI/LLM-powered tweet generation
- `trump_tweets_cleaned.csv` - Historic tweets dataset (88,449 tweets)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_predictor.txt
```

### 2. Run the Predictor

```bash
python main_tweet_predictor.py
```

The script will guide you through:
1. Choosing a model (OpenAI GPT-4, Anthropic Claude, or template-based)
2. Selecting news source (NewsAPI, RSS, or mock data)
3. Generating predictions

### 3. No API Key? No Problem!

You can run the system without any API keys:
- Select "local" for model type (template-based generation)
- Select "3" for mock news data
- Get instant predictions without external services

## Model Options

### OpenAI GPT-4 (Best Quality)
- Requires: OpenAI API key from https://platform.openai.com/
- Cost: ~$0.01-0.03 per prediction batch
- Quality: Excellent, most realistic predictions

### Anthropic Claude (High Quality)
- Requires: Anthropic API key from https://console.anthropic.com/
- Cost: ~$0.01-0.03 per prediction batch
- Quality: Excellent, very natural language

### Template-Based (Free)
- Requires: Nothing! Works offline
- Cost: Free
- Quality: Basic but captures general style

## News Sources

### NewsAPI (Recommended)
- Get free API key: https://newsapi.org/
- 100 requests/day on free tier
- Real current news headlines

### RSS Feeds (Free Alternative)
- No API key needed
- Uses public RSS feeds from CNN, BBC, Politico, etc.
- Requires `feedparser` package

### Mock Data (For Testing)
- No setup required
- Uses realistic but fictional news
- Good for testing the system

## Example Usage

### Interactive Mode
```bash
python main_tweet_predictor.py
```

### Programmatic Use

```python
from fetch_news import get_current_news, format_news_for_context
from predict_tweets import TweetPredictor

# Get news
news = get_current_news(api_key=None, use_rss=True)
news_context = format_news_for_context(news)

# Create predictor
predictor = TweetPredictor(model_type='openai')

# Generate predictions
predictions = predictor.predict(
    news_context=news_context,
    api_key='your-openai-api-key',
    num_tweets=5
)

# Display
for i, pred in enumerate(predictions, 1):
    print(f"\n=== TWEET {i} ===")
    print(pred['tweet'])
```

## Tweet Pattern Analysis

Run standalone to analyze patterns:

```bash
python analyze_tweet_patterns.py
```

This will:
- Analyze all 88,000+ historic tweets
- Extract common phrases, topics, capitalization patterns
- Save analysis to `tweet_patterns.json`
- Display comprehensive style summary

Example output:
```
TRUMP TWEET WRITING STYLE ANALYSIS
=====================================

LENGTH:
- Average: 347 characters
- Median: 298 characters

CAPITALIZATION:
- 73.2% of tweets use ALL CAPS
- Top ALL CAPS words: GREAT, AMERICA, PRESIDENT, TRUMP, etc.

TOPICS (% of tweets):
- achievements: 45.2%
- economy: 38.7%
- democrats: 34.5%
...
```

## News Fetching

Run standalone to test news sources:

```bash
python fetch_news.py
```

## Output

Predictions are saved to JSON files with timestamp:
```
predicted_tweets_20260221_143052.json
```

Format:
```json
{
  "timestamp": "2026-02-21T14:30:52",
  "model_type": "openai",
  "predictions": [
    {
      "tweet": "The Stock Market is at RECORD HIGHS...",
      "topic": "economy",
      "news_reference": "Stock Market Reaches New Highs"
    }
  ]
}
```

## Tips for Best Results

1. **Use Real News**: NewsAPI or RSS feeds give more relevant predictions
2. **Try Different Models**: OpenAI and Anthropic each have unique styles
3. **Generate Multiple**: Create 5-10 tweets to see variety
4. **Focus Topics**: Use `topic_focus` parameter for specific subjects
5. **Analyze First**: Run pattern analyzer before first prediction

## Customization

### Add Your Own News Sources

Edit `fetch_news.py` and add RSS feeds to `rss_urls`:

```python
rss_urls = [
    'https://your-favorite-news-site.com/rss',
    # ... more feeds
]
```

### Modify Writing Style

Edit the system prompt in `predict_tweets.py`:

```python
def create_system_prompt(self):
    prompt = """Your custom style instructions here..."""
    return prompt
```

### Change Model Parameters

In `predict_tweets.py`, adjust:
- `temperature`: 0.7-1.0 (higher = more creative)
- `max_tokens`: Token limit for responses
- `model`: "gpt-4-turbo-preview" or "gpt-3.5-turbo"

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements_predictor.txt
```

### NewsAPI "Invalid API key"
- Get new key from https://newsapi.org/
- Free tier: 100 requests/day
- Or switch to RSS/mock data

### OpenAI "Rate limit exceeded"
- Wait a minute between requests
- Use lower tier model (gpt-3.5-turbo)
- Or switch to template-based mode

### No predictions generated
- Check internet connection (for API calls)
- Verify API keys are correct
- Try template-based mode to test system
- Check error messages in output

## Privacy & Ethics

This tool is for:
- ✓ Research and analysis
- ✓ Understanding communication patterns
- ✓ Educational purposes
- ✓ Entertainment

Do NOT use for:
- ✗ Spreading misinformation
- ✗ Impersonation
- ✗ Deceptive content creation
- ✗ Political manipulation

Always clearly label AI-generated content as such.

## License

For educational and research purposes.

## Credits

Built with:
- OpenAI GPT-4
- Anthropic Claude
- NewsAPI
- Python pandas, requests, feedparser

Historic tweets data from public social media archives.
