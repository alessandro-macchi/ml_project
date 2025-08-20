"""
Hyperparameter optimization and parameter grid management.

This package centralizes hyperparameter tuning configurations:
- Pre-defined parameter grids for all model types
- Easy modification of search spaces
- Integration with core grid search functionality

Key modules:
- parameters_grid: Centralized parameter grid definitions for all models

To modify search spaces, edit the grids in parameters_grid.py.
The grids are designed for:
- Logistic Regression: learning rate, regularization, epochs
- SVM: regularization strength, max iterations
- Kernel methods: kernel parameters (gamma, degree, coef0), regularization
"""

from .parameters_grid import get_parameter_grids

__all__ = ['get_parameter_grids']
