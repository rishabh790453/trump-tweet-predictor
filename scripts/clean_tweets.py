import pandas as pd

# Read the CSV file
print("Reading CSV file...")
df = pd.read_csv('trump_tweets_combined.csv')

print(f"Original number of tweets: {len(df)}")

# Remove tweets with [Image] or [Video] content
df_cleaned = df[~df['content'].isin(['[Image]', '[Video]'])]
print(f"After removing [Image] and [Video] tweets: {len(df_cleaned)}")

# Remove duplicates
df_cleaned = df_cleaned.drop_duplicates()
print(f"After removing duplicates: {len(df_cleaned)}")

# Save to new CSV file
output_file = 'trump_tweets_cleaned.csv'
df_cleaned.to_csv(output_file, index=False)
print(f"\nCleaned CSV saved to: {output_file}")
print(f"Total tweets removed: {len(df) - len(df_cleaned)}")
