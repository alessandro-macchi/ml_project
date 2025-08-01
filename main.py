import os
from src.preprocessing import load_and_combine_data, preprocess_features
from models.logistic_regression import run_logistic_regression_experiment
from models.svm import run_svm_experiment
from models.kernel_logistic_regression import run_kernel_logistic_regression_experiment
from models.kernel_svm import run_kernel_svm_experiment
from src.utils import create_named_kernels
from src.save import save_results


def print_model_results(results, experiment_name):

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


def run_experiment(data, use_smote=False, experiment_name=""):
    print(f"\n{'=' * 70}")
    print(f"🧪 EXPERIMENT: {experiment_name}")
    print(f"{'=' * 70}")

    X_train, X_test, y_train, y_test = preprocess_features(data, apply_smote=use_smote)
    print(f"✅ Data preprocessing completed")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    results = {}

    # LOGISTIC REGRESSION
    print(f"\n🔹 Running Logistic Regression...")
    lr_param_grid = {
        'learning_rate': [0.08, 0.1, 0.12, 0.15],
        'regularization_strength': [0.005, 0.01, 0.02, 0.05],
        'epochs': [1000, 1200, 1500]
    }

    lr_results = run_logistic_regression_experiment(X_train, y_train, X_test, y_test, lr_param_grid)
    results.update(lr_results)

    # SVM
    print(f"\n🔹 Running SVM...")
    svm_param_grid = {
        'lambda_': [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2],
        'max_iter': [1200, 1500, 2000, 3000],
    }

    svm_results = run_svm_experiment(X_train, y_train, X_test, y_test, svm_param_grid)
    results.update(svm_results)

    # KERNEL LOGISTIC REGRESSION
    print(f"\n🔹 Running Kernel Logistic Regression...")
    klr_param_grid = {
        "kernel": create_named_kernels(gamma_values=[0.1, 0.12, 0.15], degree_values=[], coef0_values=[]),
        "lambda_": [0.005, 0.01],
        "epochs": [500, 600]
    }

    klr_results = run_kernel_logistic_regression_experiment(X_train, y_train, X_test, y_test, klr_param_grid)
    results.update(klr_results)

    # KERNEL SVM
    print(f"\n🔹 Running Kernel SVM...")
    ksvm_param_grid = {
        "kernel": create_named_kernels(gamma_values=[0.1, 0.15], degree_values=[2, 3], coef0_values=[0.5, 1, 1.5]),
        "lambda_": [0.005, 0.01, 0.05],
        "max_iter": [1000, 1500]
    }

    ksvm_results = run_kernel_svm_experiment(X_train, y_train, X_test, y_test, ksvm_param_grid)
    results.update(ksvm_results)

    print_model_results(results, experiment_name)

    save_results(results, experiment_name)

    return results


def main():
    print("🍷 WINE QUALITY CLASSIFICATION")
    print("=" * 80)

    red_path = os.path.join("data", "winequality-red.csv")
    white_path = os.path.join("data", "winequality-white.csv")

    data = load_and_combine_data(red_path, white_path)

    print("\n🚀 Starting experiments...")

    smote_results = run_experiment(data, use_smote=True, experiment_name="Wine Classification with SMOTE Oversampling")

if __name__ == "__main__":
    main()