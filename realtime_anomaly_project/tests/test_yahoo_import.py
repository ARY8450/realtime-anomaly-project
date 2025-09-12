def test_yahoo_importable():
    """Smoke test: importing the yahoo ingestion module should not raise and should expose basic symbols.
    This catches syntax/indentation/import-time errors early.
    """
    import importlib

    mod = importlib.import_module("realtime_anomaly_project.data_ingestion.yahoo")

    # basic sanity checks
    assert hasattr(mod, "data_storage")
    assert isinstance(mod.data_storage, dict)
    assert hasattr(mod, "fetch_intraday")
    assert callable(getattr(mod, "fetch_intraday"))
