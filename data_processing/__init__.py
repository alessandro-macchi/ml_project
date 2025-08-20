"""
Data processing pipeline for machine learning workflows.

This package handles all aspects of data preparation:
- Loading and combining datasets
- Feature preprocessing (scaling, transformation)
- Train-test splitting with stratification
- SMOTE-like synthetic sample generation for class balancing

Key modules:
- loaders: Dataset loading and combination utilities
- preprocessing: Feature scaling, transformation, and SMOTE augmentation
- splitting: Custom train-test split with stratification support

The preprocessing pipeline follows best practices:
1. Split data first (avoid data leakage)
2. Apply transformations fitted on training data only
3. Optionally apply SMOTE for class balancing
"""

from .loaders import load_and_combine_data
from .preprocessing import log_transform, preprocess_features
from .splitting import custom_train_test_split, generate_synthetic_samples

__all__ = [
    'load_and_combine_data',
    'log_transform', 'preprocess_features',
    'custom_train_test_split', 'generate_synthetic_samples'
]
