from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load pre-trained FinBERT model and tokenizer
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Load sentiment pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    """
    Analyze sentiment using the FinBERT model.
    Returns sentiment label (positive/neutral/negative) and the associated score.
    """
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

if __name__ == "__main__":
    # Test sentiment analysis
    sample_text = "The stock market is looking very optimistic today."
    label, score = analyze_sentiment(sample_text)
    print(f"Sentiment: {label}, Score: {score}")
