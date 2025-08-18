from src.utils import create_named_kernels
from models.logistic.base_logistic import run_logistic_regression_experiment
from models.svm.base_svm import run_svm_experiment
from models.logistic.kernel_logistic import run_kernel_logistic_regression_experiment
from models.svm.kernel_svm import run_kernel_svm_experiment

def get_parameter_grids():
    """
    Choose different values for different models' hyperparameters.
    To try different grids, just change the values here.
    """
    return {
        'lr': {
            'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.12, 0.15],
            'regularization_strength': [0.005, 0.01, 0.02, 0.05],
            'epochs': [1000, 1200, 1500]
        },
        'svm': {
            'lambda_': [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2],
            'max_iter': [1200, 1500, 2000, 3000, 4000, 5000],
        },
        'klr': {
            "kernel": create_named_kernels(gamma_values=[0.1, 0.15, 0.2], degree_values=[2, 3], coef0_values=[0.5, 1]), #gamma: 0.12, degree: 4, coef0: 1.5
            "lambda_": [0.005, 0.01], # 0.001
            "epochs": [500, 600] # 1000
        },
        'ksvm': {
            "kernel": create_named_kernels(gamma_values=[0.15, 0.2], degree_values=[2, 3], coef0_values=[0.5, 1]), #gamma: 0.1, 0.3 degree: 4, coef0: 1.5
            "lambda_": [0.0005, 0.001, 0.005], #0.01, 0.05
            "max_iter": [2000, 3000], #1000, 1500
        }
    }

def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all models with their respective hyperparameter grids

    Returns:
        tuple: (results_dict, trained_models_dict)
    """
    print(f"\n{'=' * 70}")
    print("🚀 TRAINING ALL MODELS")
    print(f"{'=' * 70}")

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

def print_model_results(results, experiment_name):
    """Print formatted results for all models"""
    print(f"\n{'=' * 70}")
    print(f"📊 RESULTS SUMMARY: {experiment_name}")
    print(f"{'=' * 70}")

    for model_name, metrics in results.items():
        print(f"\n🔹 {model_name}:")

        if isinstance(metrics, dict) and 'accuracy' in metrics:
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"   Balanced Accuracy: {metrics.get('balanced_accuracy', 'N/A'):.4f}")
            print(f"   Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"   Recall: {metrics.get('recall', 'N/A'):.4f}")
            print(f"   F1-Score: {metrics.get('f1', 'N/A'):.4f}")
        else:
            print(f"   Raw result: {metrics}")