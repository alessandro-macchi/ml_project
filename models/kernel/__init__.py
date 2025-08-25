try:
    from .kernel_logistic import KernelLogisticRegression, run_kernel_logistic_regression_experiment
    from .kernel_svm import KernelPegasosSVM, run_kernel_svm_experiment

    __all__ = [
        'KernelLogisticRegression', 'KernelPegasosSVM'
    ]
except ImportError:
    __all__ = []
