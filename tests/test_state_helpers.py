from realtime_anomaly_project.app import state


def test_state_roundtrip():
    state.ensure_initialized()
    state.set_selected_ticker("TEST1")
    assert state.get_selected_ticker() == "TEST1"
    state.set_portfolio(["aaa", "BBB", "aaa"])  # duplicates & case mix
    pf = state.get_portfolio()
    assert pf == ["AAA", "BBB"]
