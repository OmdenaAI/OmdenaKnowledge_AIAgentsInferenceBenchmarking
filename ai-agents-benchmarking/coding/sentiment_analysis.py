# filename: sentiment_analysis.py

import re

review = "'this product is okay. it's not bad.'".lower().translate({ord(c): None for c in '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'})

# Remove punctuation marks and special characters
review = re.sub(r'[^\w\s]', '', review)

positive_words = ["amazing", "love", "great"]
negative_words = ["bad", "hate"]

positive_count = sum(1 for word in review.split() if word in positive_words)
negative_count = sum(1 for word in review.split() if word in negative_words)

print("Sentiment Analysis:")
print(f"Positive Count: {positive_count}")
print(f"Negative Count: {negative_count}")

if positive_count > negative_count:
    print("Overall Sentiment: Positive")
elif positive_count < negative_count:
    print("Overall Sentiment: Negative")
else:
    print("Overall Sentiment: Neutral")