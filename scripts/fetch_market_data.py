import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz

print("Installing yfinance if needed...")
print("If this fails, run: pip install yfinance")

# Read the finance tweets CSV
print("\nReading finance tweets CSV...")
df = pd.read_csv('trump_tweets_finance.csv')

# Convert timestamp to datetime with timezone awareness
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

# Define the date range for stock data
min_date = df['timestamp'].min()
max_date = df['timestamp'].max()

print(f"\nTweet date range: {min_date} to {max_date}")
print(f"Total finance tweets: {len(df)}")

# Download stock market data for major indices
print("\nDownloading stock market data...")
print("This may take a few minutes...")

# Get data for Dow Jones (^DJI), S&P 500 (^GSPC), and NASDAQ (^IXIC)
# Using 1-minute intervals (note: Yahoo Finance only provides recent 1-min data for ~7 days)
# For older data, we'll use daily data

tickers = {
    '^DJI': 'Dow',
    '^GSPC': 'SP500',
    '^IXIC': 'NASDAQ'
}

# Get daily data for the entire period
stock_data_daily = {}
for ticker, name in tickers.items():
    print(f"Fetching {name} ({ticker})...")
    try:
        data = yf.download(ticker, start=min_date.date(), end=max_date.date() + timedelta(days=1), 
                          interval='1d', progress=False)
        stock_data_daily[ticker] = data
        print(f"  Got {len(data)} days of data")
    except Exception as e:
        print(f"  Error: {e}")

# Try to get intraday data for recent tweets (last 7 days)
recent_cutoff = datetime.now(pytz.UTC) - timedelta(days=7)
df_recent = df[df['timestamp'] > recent_cutoff]

print(f"\nRecent tweets (last 7 days): {len(df_recent)}")

stock_data_intraday = {}
if len(df_recent) > 0:
    for ticker, name in tickers.items():
        print(f"Fetching intraday data for {name}...")
        try:
            data = yf.download(ticker, period='7d', interval='1m', progress=False)
            stock_data_intraday[ticker] = data
            print(f"  Got {len(data)} minutes of data")
        except Exception as e:
            print(f"  Error: {e}")

# Match tweets to stock data
print("\nMatching tweets to stock market data...")

# Create columns for stock data
results = []

# For each tweet, find the closest stock data
for idx, row in df.iterrows():
    tweet_time = row['timestamp']
    tweet_date = tweet_time.date()
    
    result_row = {}
    
    # Use intraday data if available and recent, otherwise use daily data
    for ticker in tickers.keys():
        close_val = None
        change_1d = None
        change_pct_1d = None
        
        if ticker in stock_data_intraday and len(stock_data_intraday[ticker]) > 0:
            # Try intraday first
            intraday = stock_data_intraday[ticker]
            # Find closest time
            time_diffs = abs(intraday.index - tweet_time)
            if len(time_diffs) > 0:
                closest_idx = time_diffs.argmin()
                if time_diffs[closest_idx] < timedelta(hours=1):  # Within 1 hour
                    closest_data = intraday.iloc[closest_idx]
                    close_val = closest_data['Close']
                    
                    # Get data 1 day later for comparison
                    future_time = tweet_time + timedelta(days=1)
                    future_diffs = abs(intraday.index - future_time)
                    if len(future_diffs) > 0:
                        future_idx = future_diffs.argmin()
                        if future_diffs[future_idx] < timedelta(hours=1):
                            future_close = intraday.iloc[future_idx]['Close']
                            change_1d = future_close - closest_data['Close']
                            change_pct_1d = (change_1d / closest_data['Close']) * 100
        
        # Fall back to daily data
        if close_val is None and ticker in stock_data_daily:
            daily = stock_data_daily[ticker]
            if len(daily) > 0:
                # Find the date in the index
                daily_dates = daily.index.date
                if tweet_date in daily_dates:
                    day_data = daily[daily.index.date == tweet_date].iloc[0]
                    close_val = day_data['Close']
                    
                    # Get next trading day
                    next_day = tweet_date + timedelta(days=1)
                    attempts = 0
                    while attempts < 5:  # Try up to 5 days to find next trading day
                        if next_day in daily_dates:
                            next_data = daily[daily.index.date == next_day].iloc[0]
                            next_close = next_data['Close']
                            change_1d = next_close - day_data['Close']
                            change_pct_1d = (change_1d / day_data['Close']) * 100
                            break
                        next_day = next_day + timedelta(days=1)
                        attempts += 1
        
        # Only store percentage change
        result_row[f'{tickers[ticker]}_pct'] = change_pct_1d
    
    results.append(result_row)

# Merge results back into dataframe
results_df = pd.DataFrame(results, index=df.index)
df = pd.concat([df, results_df], axis=1)

# Keep only essential columns
columns_to_keep = ['timestamp', 'platform', 'content', 'retweets', 'likes', 'url', 
                   'Dow_pct', 'SP500_pct', 'NASDAQ_pct']
df = df[columns_to_keep]

# Save enhanced dataset
output_file = 'trump_tweets_finance_with_market_data.csv'
df.to_csv(output_file, index=False)
print(f"\nEnhanced dataset saved to: {output_file}")

# Statistics
print("\n=== Statistics ===")
tweets_with_data = df[df['Dow_pct'].notna()]
print(f"Tweets with stock market data: {len(tweets_with_data)} / {len(df)}")

if len(tweets_with_data) > 0:
    print("\n=== Sample Tweets with Market Data ===")
    sample_df = tweets_with_data.head(5).copy()
    for idx, row in sample_df.iterrows():
        print(f"\n{'='*80}")
        print(f"Date: {row['timestamp']}")
        print(f"\nContent: {str(row['content'])[:150]}...")
        print(f"\nNext Day % Change:")
        
        dow_pct = row['Dow_pct']
        if pd.notna(dow_pct):
            print(f"  Dow:     {float(dow_pct):>7.2f}%")
        
        sp_pct = row['SP500_pct']
        if pd.notna(sp_pct):
            print(f"  SP500:   {float(sp_pct):>7.2f}%")
        
        nasdaq_pct = row['NASDAQ_pct']
        if pd.notna(nasdaq_pct):
            print(f"  NASDAQ:  {float(nasdaq_pct):>7.2f}%")
