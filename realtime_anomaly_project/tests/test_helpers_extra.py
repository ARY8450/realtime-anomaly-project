import pandas as pd
from realtime_anomaly_project.app.helpers import safe_iloc_last, parse_news_item


def test_safe_iloc_last_with_dataframe():
    df = pd.DataFrame([{'a': 1}, {'a': 2}])
    last = safe_iloc_last(df)
    assert isinstance(last, dict)
    assert last['a'] == 2


def test_safe_iloc_last_empty_or_none():
    assert safe_iloc_last(pd.DataFrame()) is None
    assert safe_iloc_last(None) is None


def test_parse_news_item_full():
    item = [123, 'AAPL', '2025-01-01T00:00:00Z', 'Title', 'Summary', 'Tech', 'POS', 0.9, 'http://example.com']
    parsed = parse_news_item(item)
    assert parsed['id'] == 123
    assert parsed['ticker'] == 'AAPL'
    assert parsed['title'] == 'Title'
    assert parsed['summary'] == 'Summary'
    assert parsed['sentiment_score'] == 0.9


def test_parse_news_item_short():
    item = ['only-title']
    parsed = parse_news_item(item)
    assert parsed['title'] == 'only-title'
    assert parsed['ticker'] is None
