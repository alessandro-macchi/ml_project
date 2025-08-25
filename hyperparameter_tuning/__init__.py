"""
To modify search spaces, edit the grids in parameters_grid.py.
The grids are designed for:
- Logistic Regression: learning rate, regularization, epochs
- SVM: regularization strength, max iterations
- Kernel methods: kernel parameters (gamma, degree, coef0), regularization and epochs/max_iter
"""

from .parameters_grid import get_parameter_grids

__all__ = ['get_parameter_grids']
