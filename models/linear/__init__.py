"""
Linear classification models implemented from scratch.

This subpackage provides efficient implementations of fundamental linear models:
- Logistic Regression with regularization and margin-based loss
- Linear SVM using the Pegasos algorithm for efficiency

Key features:
- Support for {-1, +1} binary classification
- Regularization to prevent overfitting
- Numerically stable implementations
- Consistent API design

Key modules:
- logistic_regression: LogisticRegressionScratch with L2 regularization
- svm: SVMClassifierScratch using Pegasos algorithm
"""

try:
    from .logistic_regression import LogisticRegressionScratch, run_logistic_regression_experiment
    from .svm import SVMClassifierScratch, run_svm_experiment

    __all__ = [
        'LogisticRegressionScratch', 'run_logistic_regression_experiment',
        'SVMClassifierScratch', 'run_svm_experiment'
    ]
except ImportError:
    __all__ = []
