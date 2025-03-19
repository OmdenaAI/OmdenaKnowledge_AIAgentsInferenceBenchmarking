# filename: analyze_sentiment.py
import nltk

nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

review = 'This product is amazing! I love it so much.'
sentiment_scores = sia.polarity_scores(review)

print("Sentiment Scores:")
print(f"Positive: {sentiment_scores['pos']}")
print(f"Negative: {sentiment_scores['neg']}")
print(f"Neutral: {sentiment_scores['neu']}")