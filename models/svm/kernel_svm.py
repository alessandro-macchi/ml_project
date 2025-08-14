from src.metrics import comprehensive_evaluation
from src.grid_search import grid_search
import numpy as np
from src.utils import sigmoid


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

    def __init__(self, kernel, lambda_=0.01, max_iter=1000, verbose=False):
        self.kernel = kernel
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

        # Convert inputs to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.array(X)

        if hasattr(y, 'values'):
            y = y.values
        else:
            y = np.array(y)

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
                    self.kernel(sv, x_t)
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

        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.array(X)

        # Convert single sample to 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)

        decision_values = []

        for x in X:
            # Compute Σ αᵢ yᵢ K(xᵢ, x)
            kernel_values = np.array([
                self.kernel(sv, x)
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


def run_kernel_svm_experiment(X_train, y_train, X_test, y_test, param_grid):
    print("🔍 Grid search for Kernel SVM (Pegasos)...")
    best_params, best_score = grid_search(X_train, y_train, KernelPegasosSVM, param_grid)
    print(f"✅ Best KSVM params: {best_params}, CV Accuracy: {best_score:.4f}")

    model = KernelPegasosSVM(
        kernel=best_params["kernel"],
        lambda_=best_params["lambda_"],
        max_iter=best_params["max_iter"]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"📊 Number of support vectors: {len(model.support_vectors)}")

    results = {'ksvm_custom': comprehensive_evaluation(y_test, preds, "Kernel SVM (Pegasos)")}

    return results, model