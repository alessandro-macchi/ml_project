"""
Core utilities for machine learning operations.

This package provides fundamental components used across the entire ML pipeline:
- Cross-validation and grid search functionality
- Kernel functions (RBF, Polynomial) with named wrappers
- Mathematical utilities (sigmoid, distance functions)
- Comprehensive evaluation metrics for classification

Key modules:
- cross_validation: K-fold CV and grid search with progress tracking
- kernels: Kernel functions and utilities for kernel methods
- math_utils: Mathematical functions with numerical stability
- metrics: Comprehensive model evaluation metrics
"""

from .cross_validation import cross_validate, grid_search
from .kernels import rbf_kernel, polynomial_kernel, NamedKernel, create_named_kernels
from .math_utils import sigmoid, euclidean_distance
from .metrics import comprehensive_evaluation

__all__ = [
    'cross_validate', 'grid_search',
    'rbf_kernel', 'polynomial_kernel', 'NamedKernel', 'create_named_kernels',
    'sigmoid', 'euclidean_distance',
    'comprehensive_evaluation'
]
