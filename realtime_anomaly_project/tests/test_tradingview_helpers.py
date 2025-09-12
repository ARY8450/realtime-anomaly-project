from realtime_anomaly_project.app.helpers import tradingview_symbol, tradingview_embed_html


def test_tradingview_symbol_ns():
    assert tradingview_symbol('RELIANCE.NS') == 'NSE:RELIANCE'
    assert tradingview_symbol('reliance.ns') == 'NSE:RELIANCE'


def test_tradingview_symbol_plain():
    assert tradingview_symbol('AAPL') == 'AAPL'


def test_tradingview_embed_contains_symbol():
    html = tradingview_embed_html('RELIANCE.NS', width=600, height=300)
    assert 'NSE:RELIANCE' in html
    assert 'height:300' in html or 'height: 300' in html
