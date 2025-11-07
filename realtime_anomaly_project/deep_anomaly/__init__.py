"""
Deep Anomaly Detection Module
Contains Transformer AutoEncoder and advanced anomaly detection implementations
"""

from .transformer_ae import (
    TransformerAutoencoder,
    train_model,
    compute_reconstruction_error,
    detect_anomalies,
    fit_transformer,
    compute_deep_anomalies
)

from .advanced_anomaly_detector import AdvancedAnomalyDetector

__all__ = [
    'TransformerAutoencoder',
    'AdvancedAnomalyDetector',
    'train_model',
    'compute_reconstruction_error',
    'detect_anomalies',
    'fit_transformer',
    'compute_deep_anomalies'
]
