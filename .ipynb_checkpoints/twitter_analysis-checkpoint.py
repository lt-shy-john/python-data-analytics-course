import pandas as pd
import re
from collections import Counter

def find_mentions(tweets):
    tweets_df['mentions'] = tweets_df['text'].str.contains('[@]\w+', regex=True)

tweets_df = pd.read_csv('tweets.csv')
tweets_df.pop('Unnamed: 0')
# print(tweets_df.keys())
print(tweets_df.head(5))
# print(tweets_df['text'].head(5))

print('Frequent words')
print(Counter(" ".join(tweets_df["text"]).split()).most_common(100))

tweetsNSW_df = tweets_df[tweets_df['text'].str.contains("NSW")]
tweetsNSW_df = tweetsNSW_df.append(tweets_df[tweets_df['text'].str.contains("nsw")])
# print(len(tweetsNSW_df))

tweetsQLD_df = tweets_df[tweets_df['text'].str.contains("qld")]
tweetsQLD_df = tweetsQLD_df.append(tweets_df[tweets_df['text'].str.contains("QLD")])
# print(len(tweetsQLD_df))

print('===== LOCATION =====')
print('=== NSW ===')
print(tweetsNSW_df['author_location'])
print('\n===QLD===')
print('=== NSW ===')
print(tweetsQLD_df['author_location'])

find_mentions(tweets_df)
# print(tweets_df.mentions)
