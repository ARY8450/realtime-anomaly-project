import re
from realtime_anomaly_project.app.helpers import tradingview_embed_html


def test_tradingview_full_width_breakout():
    html = tradingview_embed_html("AAPL", width=0, height=400)
    # Expect 100vw wrapper and iframe present
    assert "100vw" in html
    assert "iframe" in html.lower()
    # Ensure symbol appears (either direct or transformed)
    assert "AAPL" in html or "AAPL".lower() in html.lower()
