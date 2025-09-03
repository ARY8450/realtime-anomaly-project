try:
    import realtime_anomaly_project.app.dashboard as dash
    print('import ok')
except Exception as e:
    print('import failed', e)
    raise
