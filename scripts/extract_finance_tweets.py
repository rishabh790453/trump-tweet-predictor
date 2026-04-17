import pandas as pd
import re
from datetime import datetime, timedelta

# Read the cleaned CSV file
print("Reading cleaned CSV file...")
df = pd.read_csv('trump_tweets_cleaned.csv')

print(f"Total tweets: {len(df)}")

# Define finance-related keywords to search for
finance_keywords = [
    'dow', 'dow jones', 's&p', 'stock market', 'stocks', 'nasdaq',
    'wall street', 'trading', 'market', 'investor', 'investment',
    '401k', 'portfolio', 'ticker', 'equity', 'equities',
    'bull market', 'bear market', 'rally', 'gains', 'losses'
]

# Create a pattern to search for any of these keywords (case-insensitive)
pattern = '|'.join([re.escape(keyword) for keyword in finance_keywords])

# Filter tweets containing finance-related terms
df_finance = df[df['content'].str.contains(pattern, case=False, na=False, regex=True)]

print(f"Finance-related tweets found: {len(df_finance)}")

# Convert timestamp to datetime
df_finance['timestamp'] = pd.to_datetime(df_finance['timestamp'])

# Sort by timestamp
df_finance = df_finance.sort_values('timestamp', ascending=False)

# Save to new CSV
output_file = 'trump_tweets_finance.csv'
df_finance.to_csv(output_file, index=False)
print(f"\nFinance tweets saved to: {output_file}")

# Display sample of the tweets
print("\n=== Sample Finance Tweets ===")
for idx, row in df_finance.head(10).iterrows():
    print(f"\nDate: {row['timestamp']}")
    print(f"Platform: {row['platform']}")
    print(f"Content: {row['content'][:200]}..." if len(row['content']) > 200 else f"Content: {row['content']}")
    print("-" * 80)
