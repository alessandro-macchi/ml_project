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
    """
    Kernel SVM using the Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) algorithm.

    Why Pegasos for Kernel SVM?

    1. **Efficiency**: Pegasos is a stochastic sub-gradient descent method that converges
       in O(1/λε) iterations, making it very efficient for large datasets.

    2. **Online Learning**: Pegasos processes one example at a time, making it suitable
       for online/streaming scenarios and memory-constrained environments.

    3. **Kernelization**: The algorithm naturally extends to kernels by maintaining
       a sparse representation using only "support vectors" (misclassified examples).

    4. **Theoretical Guarantees**: Pegasos has strong convergence guarantees and
       achieves O(1/√T) convergence rate in the number of iterations T.

    5. **Sparsity**: The algorithm automatically maintains sparsity by only keeping
       examples that violate the margin, leading to natural support vector selection.

    The kernel version works by:
    - Maintaining a set of support vectors (examples that violated the margin)
    - Computing decision function as: f(x) = Σ αᵢ yᵢ K(xᵢ, x) where αᵢ are coefficients
    - Updating coefficients based on margin violations using sub-gradient descent
    """

    def __init__(self, kernel_fn, lambda_=0.01, max_iter=1000, verbose=False):
        self.kernel_fn = kernel_fn
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.verbose = verbose

        # Support vectors and their coefficients
        self.support_vectors = []
        self.support_labels = []
        self.alphas = []  # Coefficients for each support vector

    def fit(self, X, y, max_iter=None):
        """
        Train the kernel SVM using the Pegasos algorithm.

        Args:
            X: Training data (n_samples, n_features)
            y: Training labels {0, 1} (converted internally to {-1, +1})
            max_iter: Maximum number of iterations (overrides constructor)
        """
        if max_iter is not None:
            self.max_iter = max_iter

        # Convert labels to {-1, +1} format required by SVM
        y_transformed = np.where(y == 0, -1, y)

        if set(np.unique(y_transformed)) != {-1, 1}:
            raise ValueError("Pegasos requires binary classification with labels in {-1, 1}")

        n_samples = X.shape[0]

        # Clear previous training data
        self.support_vectors = []
        self.support_labels = []
        self.alphas = []

        if self.verbose:
            print(f"     🚀 Training Kernel Pegasos SVM for {self.max_iter} iterations...")

        # Track some statistics for debugging
        margin_violations = 0

        for t in range(1, self.max_iter + 1):
            # Sample a random training example
            i = np.random.randint(0, n_samples)
            x_t = X[i]
            y_t = y_transformed[i]  # Fixed: use correct label

            # Compute decision function value: f(x) = Σ αᵢ yᵢ K(xᵢ, x)
            if len(self.support_vectors) == 0:
                decision_value = 0.0
            else:
                # Compute kernel values between current example and all support vectors
                kernel_values = np.array([
                    self.kernel_fn(sv, x_t)
                    for sv in self.support_vectors
                ])
                decision_value = np.sum(
                    np.array(self.alphas) * np.array(self.support_labels) * kernel_values
                )

            # Learning rate schedule: η_t = 1/(λt)
            eta_t = 1.0 / (self.lambda_ * t)

            # Check margin violation: yᵢ f(xᵢ) < 1
            margin = y_t * decision_value

            if margin < 1:
                # Margin violation: add this example as a support vector
                self.support_vectors.append(x_t.copy())
                self.support_labels.append(y_t)
                self.alphas.append(eta_t)  # α_new = η_t
                margin_violations += 1

            # Decay existing coefficients: α_i ← (1 - η_t λ) α_i
            # This is crucial for preventing coefficient explosion
            decay_factor = max(0, 1 - eta_t * self.lambda_)  # Ensure non-negative
            self.alphas = [alpha * decay_factor for alpha in self.alphas]

            # Optional: Remove support vectors with very small coefficients for efficiency
            if t % 500 == 0:
                self._prune_support_vectors(threshold=1e-8)

            # Progress reporting
            if self.verbose and t % max(1, self.max_iter // 10) == 0:
                n_sv = len(self.support_vectors)
                violation_rate = margin_violations / t
                print(f"       Iteration {t:4d}/{self.max_iter}: {n_sv} SVs, {violation_rate:.3f} violation rate")

        if self.verbose:
            print(
                f"     ✅ Training completed. Final: {len(self.support_vectors)} SVs, {margin_violations} total violations")

        # Final pruning
        self._prune_support_vectors(threshold=1e-10)

    def _prune_support_vectors(self, threshold=1e-8):
        """Remove support vectors with coefficients below threshold"""
        if not self.alphas:
            return

        # Find indices of significant support vectors
        significant_indices = [
            i for i, alpha in enumerate(self.alphas)
            if abs(alpha) > threshold
        ]

        # Keep only significant support vectors
        if significant_indices:
            self.support_vectors = [self.support_vectors[i] for i in significant_indices]
            self.support_labels = [self.support_labels[i] for i in significant_indices]
            self.alphas = [self.alphas[i] for i in significant_indices]
        # If all coefficients are too small, keep at least one
        elif self.support_vectors:
            # Keep the support vector with largest coefficient
            max_idx = np.argmax([abs(alpha) for alpha in self.alphas])
            self.support_vectors = [self.support_vectors[max_idx]]
            self.support_labels = [self.support_labels[max_idx]]
            self.alphas = [self.alphas[max_idx]]

    def decision_function(self, X):
        """
        Compute decision function values for input samples.

        Args:
            X: Input samples (n_samples, n_features)

        Returns:
            Decision values (n_samples,)
        """
        if not self.support_vectors:
            # No support vectors: return small negative values (predict negative class)
            return np.full(X.shape[0], -0.1)

        # Convert single sample to 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)

        decision_values = []

        for x in X:
            # Compute Σ αᵢ yᵢ K(xᵢ, x)
            kernel_values = np.array([
                self.kernel_fn(sv, x)
                for sv in self.support_vectors
            ])

            decision_value = np.sum(
                np.array(self.alphas) * np.array(self.support_labels) * kernel_values
            )

            decision_values.append(decision_value)

        return np.array(decision_values)

    def predict(self, X):
        """
        Predict class labels for input samples.

        Args:
            X: Input samples (n_samples, n_features)

        Returns:
            Predicted labels {0, 1} (converted from {-1, +1})
        """
        decision_values = self.decision_function(X)

        # Convert {-1, +1} predictions back to {0, 1}
        predictions = np.where(decision_values >= 0, 1, 0)

        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities (approximation using sigmoid).

        Note: This is an approximation since SVM doesn't naturally output probabilities.
        """
        decision_values = self.decision_function(X)

        # Use sigmoid to convert decision values to probabilities
        # Scale by a factor to get reasonable probabilities
        probabilities = sigmoid(decision_values)

        return probabilities



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