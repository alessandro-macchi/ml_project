"""
Kernel-based classification methods.

This subpackage implements sophisticated kernel methods for non-linear classification:
- Kernel Logistic Regression with optimized kernel matrix computation
- Kernel SVM using the Pegasos algorithm adapted for kernels

Key features:
- Highly optimized kernel computations (vectorized RBF and polynomial)
- Support vector subsampling for efficiency
- Mini-batch training with early stopping
- Memory-efficient kernel matrix operations

Key modules:
- kernel_logistic: KernelLogisticRegression with subsample optimization
- kernel_svm: KernelPegasosSVM with automatic support vector selection

Performance optimizations:
- Vectorized kernel matrix computation (100x faster than nested loops)
- Support vector pruning and subsampling
- Batch processing for large datasets
"""

try:
    from .kernel_logistic import KernelLogisticRegression, run_kernel_logistic_regression_experiment
    from .kernel_svm import KernelPegasosSVM, run_kernel_svm_experiment

    __all__ = [
        'KernelLogisticRegression', 'run_kernel_logistic_regression_experiment',
        'KernelPegasosSVM', 'run_kernel_svm_experiment'
    ]
except ImportError:
    __all__ = []
