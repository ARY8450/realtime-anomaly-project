from realtime_anomaly_project.app import helpers


def test_format_mktcap():
    assert helpers._format_mktcap(1_500_000_000_000) == '1.50T'
    assert helpers._format_mktcap(2_500_000_000) == '2.50B'
    assert helpers._format_mktcap(3_500_000) == '3.50M'
    assert helpers._format_mktcap('notanumber') == 'notanumber'


def test_safe_float_and_optional():
    assert helpers.safe_float('3.14') == 3.14
    assert helpers.safe_float(None, default=1.2) == 1.2
    assert helpers.safe_optional_float('5.5') == 5.5
    assert helpers.safe_optional_float('bad') is None
    assert helpers.safe_optional_float(None) is None
