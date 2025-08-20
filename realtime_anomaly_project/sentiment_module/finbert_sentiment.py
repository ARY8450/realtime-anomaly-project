from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Any, cast
import importlib

# Load pre-trained FinBERT model and tokenizer
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Load sentiment pipeline
# The `pipeline` function has many overloads which can confuse static type checkers.
# Cast to Any to keep runtime behavior while silencing Pylance type complaints.
# Use importlib to call `pipeline` at runtime which avoids Pylance overload/type resolution issues.
transformers_module = importlib.import_module('transformers')
pipeline_fn = getattr(transformers_module, 'pipeline')
sentiment_analyzer: Any = cast(Any, pipeline_fn("sentiment-analysis", model=model, tokenizer=tokenizer))

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
