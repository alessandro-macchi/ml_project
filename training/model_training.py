from models.linear.logistic_regression import  run_logistic_regression_experiment
from models.linear.svm import  run_svm_experiment
from models.kernel.kernel_logistic import  run_kernel_logistic_regression_experiment
from models.kernel.kernel_svm import  run_kernel_svm_experiment
from hyperparameter_tuning.parameters_grid import get_parameter_grids


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
    lr_results, lr_model = run_logistic_regression_experiment(
        X_train, y_train, X_test, y_test, param_grids['lr']
    )
    results.update(lr_results)
    trained_models['lr_custom'] = lr_model

    # SVM
    print(f"\n🔹 Running SVM...")
    svm_results, svm_model = run_svm_experiment(
        X_train, y_train, X_test, y_test, param_grids['svm']
    )
    results.update(svm_results)
    trained_models['svm_custom'] = svm_model

    # KERNEL LOGISTIC REGRESSION
    print(f"\n🔹 Running Kernel Logistic Regression...")
    klr_results, klr_model = run_kernel_logistic_regression_experiment(
        X_train, y_train, X_test, y_test, param_grids['klr']
    )
    results.update(klr_results)
    trained_models['klr_custom'] = klr_model

    # KERNEL SVM
    print(f"\n🔹 Running Kernel SVM...")
    ksvm_results, ksvm_model = run_kernel_svm_experiment(
        X_train, y_train, X_test, y_test, param_grids['ksvm']
    )
    results.update(ksvm_results)
    trained_models['ksvm_custom'] = ksvm_model

    return results, trained_models