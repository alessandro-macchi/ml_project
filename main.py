import os
from src.preprocessing import load_and_combine_data, preprocess_features
from models.logistic_regression import run_logistic_regression_experiment
from models.svm import run_svm_experiment
from models.kernel_logistic_regression import run_kernel_logistic_regression_experiment
from models.kernel_svm import run_kernel_svm_experiment
from src.utils import create_named_kernels


def run_experiment(data, use_smote=False, experiment_name=""):
    print(f"\n{'=' * 70}")
    print(f"🧪 EXPERIMENT: {experiment_name}")
    print(f"{'=' * 70}")

    X_train, X_test, y_train, y_test = preprocess_features(data, apply_smote=use_smote)

    results = {}

    # LOGISTIC REGRESSION
    lr_param_grid = {
        'learning_rate': [0.08, 0.1, 0.12, 0.15],
        'regularization_strength': [0.005, 0.01, 0.02, 0.05],
        'epochs': [1000, 1200, 1500]
    }
    results.update(run_logistic_regression_experiment(X_train, y_train, X_test, y_test, lr_param_grid))

    # SVM
    svm_param_grid = {
        'lambda_': [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2],
        'max_iter': [1200, 1500, 2000, 3000],
    } # Use C = 1/lambda for sklearn
    results.update(run_svm_experiment(X_train, y_train, X_test, y_test, svm_param_grid))

    # KERNEL LOGISTIC REGRESSION
    klr_param_grid = {
        "kernel": create_named_kernels(gamma_values=[0.1, 0.12, 0.15], degree_values=[], coef0_values=[]),
        "lambda_": [0.005, 0.01, 0.015],
        "epochs": [400, 500, 600]
    }
    results.update(run_kernel_logistic_regression_experiment(X_train, y_train, X_test, y_test, klr_param_grid))

    # KERNEL SVM
    ksvm_param_grid = {
        "kernel": create_named_kernels(gamma_values=[0.1, 0.15], degree_values=[2, 3], coef0_values=[0.5, 1, 1.5]),
        "lambda_": [0.005, 0.01, 0.05],
        "max_iter": [1000, 1500]
    } # Use C = 1/lambda for sklearn
    results.update(run_kernel_svm_experiment(X_train, y_train, X_test, y_test, ksvm_param_grid))

    return results


def main():
    print("🍷 WINE QUALITY CLASSIFICATION")
    print("=" * 80)

    red_path = os.path.join("data", "winequality-red.csv")
    white_path = os.path.join("data", "winequality-white.csv")
    data = load_and_combine_data(red_path, white_path)

    print("\n🚀 Starting experiments...")

    run_experiment(data, use_smote=True, experiment_name="SMOTE OVERSAMPLING")


if __name__ == "__main__":
    main()
