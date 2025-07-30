import os
import numpy as np
from src.preprocessing import load_and_combine_data, preprocess_features
from src.logistic_regression import LogisticRegressionScratch
from src.svm import SVMClassifierScratch
from src.kernels import KernelLogisticRegression, KernelPegasosSVM, create_named_kernels
from src.hyperparameter_tuning import grid_search
from sklearn.linear_model import LogisticRegression  # benchmark
from sklearn.svm import SVC  # benchmark
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score , confusion_matrix# evaluation


def comprehensive_evaluation(y_true, y_pred, model_name="Model"):
    """Comprehensive evaluation for imbalanced classification"""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"✅ {model_name} Results:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Balanced Accuracy: {bal_acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"   Confusion Matrix: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class KernelPegasosSVM:
    def __init__(self, lambda_param=0.01, max_iter=1000, kernel='rbf', gamma=0.05, degree=3, coef0=1):
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = None
        self.X_train = None
        self.y_train = None

    def _kernel_function(self, X1, X2):
        if self.kernel == 'rbf':
            # RBF (Gaussian) Kernel
            if X1.ndim == 1:
                X1 = X1.reshape(1, -1)
            if X2.ndim == 1:
                X2 = X2.reshape(1, -1)
            sq_dists = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
            return np.exp(-self.gamma * sq_dists)

        elif self.kernel == 'poly':
            # Polynomial Kernel
            return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree

        else:
            raise ValueError("Unsupported kernel. Choose 'rbf' or 'poly'.")

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y
        self.alpha = np.zeros(n_samples)

        for t in range(1, self.max_iter + 1):
            i = np.random.randint(0, n_samples)
            eta = 1 / (self.lambda_param * t)

            # Kernel evaluation between sample i and all training samples
            K_i = self._kernel_function(X[i], X).flatten()

            # Compute decision value
            decision = np.sum(self.alpha * y * K_i)

            # Perform update if margin violated
            if y[i] * decision < 1:
                self.alpha[i] += 1

    def project(self, X):
        K = self._kernel_function(X, self.X_train)  # Shape: (n_test, n_train)
        return np.dot(K, self.alpha * self.y_train)

    def predict(self, X):
        return np.sign(self.project(X))


def run_experiment(data, use_smote=False, experiment_name="Baseline"):
    """Run a complete experiment with or without SMOTE"""

    print(f"\n{'=' * 70}")
    print(f"🧪 EXPERIMENT: {experiment_name}")
    print(f"{'=' * 70}")

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_features(data, apply_smote=use_smote)

    results = {}

    # 4. KERNEL SVM
    print(f"\n{'=' * 50}")
    print("🔍 KERNEL SVM")
    print(f"{'=' * 50}")

    gamma_values = [0.01, 0.1]
    degree_values = []  # 2, 3
    coef0_values = []  # 0, 1
    named_kernels = create_named_kernels(gamma_values, degree_values, coef0_values)

    ksvm_param_grid = {
        "kernel": named_kernels[:3],  # Limit to first 3 kernels for speed
        "lambda_param": [0.01, 0.1],
        "max_iter": [500]
    }

    print("🔍 Grid search for Kernel SVM...")
    best_params_ksvm, best_score_ksvm = grid_search(X_train, y_train, KernelPegasosSVM, ksvm_param_grid)
    print(f"✅ Best KSVM params: {best_params_ksvm}, CV Accuracy: {best_score_ksvm:.4f}")

    # Train best model
    ksvm = KernelPegasosSVM(
        kernel=best_params_ksvm["kernel_fn"],
        lambda_param=best_params_ksvm["lambda_param"],
        max_iter=best_params_ksvm["max_iter"]
    )
    ksvm.fit(X_train, y_train)
    pred_ksvm = ksvm.predict(X_test)

    results['ksvm_scratch'] = comprehensive_evaluation(y_test, pred_ksvm, "Kernel SVM (Scratch)")
    print(f"📊 Number of support vectors: {len(ksvm.support_vectors)}")


    return results




def main():
    """Main function comparing baseline vs SMOTE approaches"""

    print("🍷 WINE QUALITY CLASSIFICATION: BASELINE vs SMOTE COMPARISON")
    print("=" * 80)

    # Load data
    red_path = os.path.join("..", "data", "winequality-red.csv")
    white_path = os.path.join("..", "data", "winequality-white.csv")
    data = load_and_combine_data(red_path, white_path)

    # Run experiments
    print("\n🚀 Starting experiments...")

    # Baseline experiment (no SMOTE)
    #results_baseline = run_experiment(data, use_smote=False, experiment_name="BASELINE (No SMOTE)")

    # SMOTE experiment
    results_smote = run_experiment(data, use_smote=True, experiment_name="SMOTE OVERSAMPLING")



if __name__ == "__main__":
    main()