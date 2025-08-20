"""
Model training pipeline and experiment management.

This package orchestrates the complete model training workflow:
- Coordination of all model training with their respective hyperparameters
- Integration of data preprocessing, training, and evaluation
- Experiment management with consistent naming and organization

Key modules:
- experiment: Main experiment runner coordinating the entire pipeline
- model_training: Individual model training with hyperparameter optimization

The training pipeline:
1. Data preprocessing (with optional SMOTE)
2. Hyperparameter optimization for each model
3. Model training with best parameters
4. Comprehensive evaluation and analysis
"""

try:
    from .experiment import run_experiment
    from .model_training import train_all_models

    __all__ = ['run_experiment', 'train_all_models']
except ImportError:
    __all__ = []
