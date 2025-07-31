import os
from src.preprocessing import load_and_combine_data, preprocess_features
from models.logistic_regression import run_logistic_regression_experiment
from models.svm import run_svm_experiment
from models.kernel_logistic_regression import run_kernel_logistic_regression_experiment
from models.kernel_svm import run_kernel_svm_experiment
from src.utils import create_named_kernels
from src.metrics import print_comparison_table


def run_experiment(data, use_smote=False, experiment_name="Baseline"):
    """Run a complete experiment with or without SMOTE"""
    print(f"\n{'=' * 70}")
    print(f"🧪 EXPERIMENT: {experiment_name}")
    print(f"{'=' * 70}")

    X_train, X_test, y_train, y_test = preprocess_features(data, apply_smote=use_smote)

    results = {}

    # LOGISTIC REGRESSION
    lr_param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "regularization_strength": [0.0, 0.01, 0.1],
        "epochs": [500, 1000]
    }
    results.update(run_logistic_regression_experiment(X_train, y_train, X_test, y_test, lr_param_grid))

    # SVM
    svm_param_grid = {
        "lambda_": [0.001, 0.01, 0.1],
        "max_iter": [500, 1000]
    }
    results.update(run_svm_experiment(X_train, y_train, X_test, y_test, svm_param_grid))

    # KERNEL LOGISTIC REGRESSION
    klr_param_grid = {
        "kernel": create_named_kernels(gamma_values=[0.01, 0.1], degree_values=[], coef0_values=[]),
        "lambda_": [0.01, 0.1],
        "epochs": [500]
    }
    results.update(run_kernel_logistic_regression_experiment(X_train, y_train, X_test, y_test, klr_param_grid))

    # KERNEL SVM
    ksvm_param_grid = {
        "kernel": create_named_kernels(gamma_values=[0.01, 0.1], degree_values=[2, 3], coef0_values=[0, 1]),
        "lambda_": [0.01, 0.1],
        "max_iter": [1000, 2000]
    }
    results.update(run_kernel_svm_experiment(X_train, y_train, X_test, y_test, ksvm_param_grid))

    return results


def main():
    print("🍷 WINE QUALITY CLASSIFICATION: BASELINE vs SMOTE COMPARISON")
    print("=" * 80)

    red_path = os.path.join("data", "winequality-red.csv")
    white_path = os.path.join("data", "winequality-white.csv")
    data = load_and_combine_data(red_path, white_path)

    print("\n🚀 Starting experiments...")

    # Baseline experiment (no SMOTE)
    results_baseline = run_experiment(data, use_smote=False, experiment_name="BASELINE (No SMOTE)")

    # SMOTE experiment
    results_smote = run_experiment(data, use_smote=True, experiment_name="SMOTE OVERSAMPLING")

    # Print detailed comparison
    print_comparison_table(results_baseline, results_smote)


if __name__ == "__main__":
    main()
