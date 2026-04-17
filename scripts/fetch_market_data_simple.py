import pandas as pd
import yfinance as yf
from datetime import timedelta

print("Reading finance tweets CSV...")
df = pd.read_csv('trump_tweets_finance.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

print(f"Total finance tweets: {len(df)}")

# Download stock data
tickers = {'^DJI': 'Dow', '^GSPC': 'SP500', '^IXIC': 'NASDAQ'}
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()

print("\nDownloading stock data (this takes a minute)...")
stock_data = {}
for ticker, name in tickers.items():
    print(f"  Fetching {name}...")
    data = yf.download(ticker, start=min_date, end=max_date + timedelta(days=10), 
                      interval='1d', progress=False)
    stock_data[ticker] = data

print("\nMatching tweets to market data...")

# For each tweet, get next day's percentage change
results = []
for idx, row in df.iterrows():
    if idx % 1000 == 0:
        print(f"  Processed {idx}/{len(df)} tweets...")
    
    tweet_date = row['timestamp'].date()
    result = {}
    
    for ticker, name in tickers.items():
        pct_change = None
        data = stock_data[ticker]
        
        # Find the tweet date in the data
        matching_dates = data.index[data.index.date == tweet_date]
        
        if len(matching_dates) > 0:
            current_idx = data.index.get_loc(matching_dates[0])
            # Get next trading day
            if current_idx + 1 < len(data):
                current_close = float(data.iloc[current_idx]['Close'])
                next_close = float(data.iloc[current_idx + 1]['Close'])
                pct_change = round(((next_close - current_close) / current_close) * 100, 2)
        
        result[f'{name}_pct'] = pct_change
    
    results.append(result)

# Add results to dataframe
results_df = pd.DataFrame(results)
df = pd.concat([df, results_df], axis=1)

# Keep only essential columns
columns = ['timestamp', 'platform', 'content', 'retweets', 'likes', 'url', 
           'Dow_pct', 'SP500_pct', 'NASDAQ_pct']
df = df[columns]

# Save
output_file = 'trump_tweets_finance_with_market_data.csv'
df.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")

# Show stats
with_data = df[df['Dow_pct'].notna()]
print(f"Tweets with market data: {len(with_data)}/{len(df)}")

# Show samples
print("\n=== Sample Tweets ===")
sample_data = with_data.head(5).to_dict('records')
for row in sample_data:
    print(f"\nDate: {row['timestamp']}")
    print(f"Content: {str(row['content'])[:100]}...")
    print(f"Next Day Changes:")
    if pd.notna(row['Dow_pct']):
        print(f"  Dow:     {row['Dow_pct']:7.2f}%")
    if pd.notna(row['SP500_pct']):
        print(f"  SP500:   {row['SP500_pct']:7.2f}%")
    if pd.notna(row['NASDAQ_pct']):
        print(f"  NASDAQ:  {row['NASDAQ_pct']:7.2f}%")
