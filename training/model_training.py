from models import LogisticRegressionScratch, SVMClassifierScratch, KernelLogisticRegression, KernelPegasosSVM
from hyperparameter_tuning import get_parameter_grids


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all models with their respective hyperparameter grids

    Returns:
        tuple: (results_dict, trained_models_dict)
    """

    results = {}
    trained_models = {}

    # Define parameter grids
    param_grids = get_parameter_grids()

    # LOGISTIC REGRESSION
    print(f"\n🔹 Running Logistic Regression...")
    lr_results, lr_model = LogisticRegressionScratch.run_logistic_regression_experiment(
        X_train, y_train, X_test, y_test, param_grids['logistic_regression']
    )
    results.update(lr_results)
    trained_models['lr_custom'] = lr_model

    # SVM
    print(f"\n🔹 Running SVM...")
    svm_results, svm_model = SVMClassifierScratch.run_svm_experiment(
        X_train, y_train, X_test, y_test, param_grids['svm']
    )
    results.update(svm_results)
    trained_models['svm_custom'] = svm_model

    # KERNEL LOGISTIC REGRESSION
    print(f"\n🔹 Running Kernel Logistic Regression...")
    klr_results, klr_model = KernelLogisticRegression.run_kernel_logistic_regression_experiment(
        X_train, y_train, X_test, y_test, param_grids['kernel_logistic_regression']
    )
    results.update(klr_results)
    trained_models['klr_custom'] = klr_model

    # KERNEL SVM
    print(f"\n🔹 Running Kernel SVM...")
    ksvm_results, ksvm_model = KernelPegasosSVM.run_kernel_svm_experiment(
        X_train, y_train, X_test, y_test, param_grids['kernel_svm']
    )
    results.update(ksvm_results)
    trained_models['ksvm_custom'] = ksvm_model

    return results, trained_models