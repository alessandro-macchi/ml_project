"""
Custom machine learning model implementations.

This package contains from-scratch implementations of classical ML algorithms:
- Linear models (Logistic Regression, SVM)
- Kernel methods (Kernel Logistic Regression, Kernel SVM with Pegasos)
- All models support hyperparameter tuning and comprehensive evaluation

Key subpackages:
- linear: Linear classification algorithms
- kernel: Kernel-based classification methods

Design principles:
- Consistent API across all models (fit/predict/predict_proba)
- Optimized implementations with numerical stability
- Support for {-1, +1} binary classification labels
- Integration with cross-validation and grid search
"""

# Import key model classes
try:
    from .linear.logistic_regression import LogisticRegressionScratch
    from .linear.svm import SVMClassifierScratch
    from .kernel.kernel_logistic import KernelLogisticRegression
    from .kernel.kernel_svm import KernelPegasosSVM

    __all__ = [
        'LogisticRegressionScratch', 'SVMClassifierScratch',
        'KernelLogisticRegression', 'KernelPegasosSVM'
    ]
except ImportError:
    __all__ = []
