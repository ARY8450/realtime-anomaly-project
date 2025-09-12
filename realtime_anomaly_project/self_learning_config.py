"""
Configuration for Self-Learning System
"""

# Target Accuracies
TARGET_ACCURACIES = {
    'primary': 0.98,      # Primary target (98%)
    'fallback': 0.96,     # Fallback target (96%)
    'minimum': 0.90       # Minimum acceptable (90%)
}

# Training Configuration
TRAINING_CONFIG = {
    'max_iterations_per_session': 50,
    'patience': 10,
    'early_stopping_threshold': 0.001,  # Stop if improvement < 0.1%
    'training_interval_hours': 12,       # Train every 12 hours
    'continuous_mode': True,
    'emergency_retrain_threshold': 0.05  # Retrain if accuracy drops by 5%
}

# Model Configuration
MODEL_CONFIG = {
    'anomaly_detection': {
        'algorithms': ['RandomForest', 'GradientBoosting', 'IsolationForest'],
        'hyperparameter_trials': 50,
        'cross_validation_folds': 5,
        'use_gpu_acceleration': True,
        'n_jobs_gpu': -1  # Use all cores when GPU available
    },
    'sentiment_analysis': {
        'algorithms': ['BERT', 'LSTM', 'RandomForest'],
        'hyperparameter_trials': 30,
        'batch_size': 32,  # Will be auto-adjusted for GPU
        'max_epochs': 50,
        'use_mixed_precision': True,  # GPU optimization
        'gradient_accumulation_steps': 2  # For large models
    },
    'trend_prediction': {
        'algorithms': ['SAC', 'TD3', 'PPO'],
        'rl_timesteps': 500000,  # Will be increased for GPU
        'evaluation_episodes': 20,
        'curriculum_levels': 5,
        'use_gpu_vectorization': True,
        'parallel_envs_gpu': 16  # More envs with GPU
    }
}

# Data Configuration
DATA_CONFIG = {
    'anomaly_synthetic_samples': 10000,
    'sentiment_samples_per_class': 1000,
    'trend_tickers_subset': 10,
    'train_test_split': 0.8,
    'data_augmentation': True
}

# Model Management
MODEL_MANAGEMENT = {
    'version_retention_days': 30,
    'ab_test_sample_size': 1000,
    'ab_test_confidence': 0.95,
    'deployment_improvement_threshold': 0.02,  # 2% improvement for deployment
    'performance_monitoring_window': 100
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'file': 'self_learning_system.log',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Paths
PATHS = {
    'models': 'production_models',
    'logs': 'logs',
    'data': 'training_data',
    'temp': 'temp',
    'backup': 'backup'
}

# System Configuration
SYSTEM_CONFIG = {
    'max_concurrent_trainings': 3,
    'memory_limit_gb': 8,
    'cpu_cores': -1,  # Use all available
    'gpu_enabled': True,
    'gpu_memory_fraction': 0.8,   # Use 80% of GPU memory
    'mixed_precision': True,      # Use mixed precision training
    'cuda_benchmark': True,       # Enable CUDA optimizations
    'auto_batch_size': True       # Automatically adjust batch sizes for GPU
}
