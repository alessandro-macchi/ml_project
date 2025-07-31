import numpy as np


def linear_kernel(X1, X2):
    return X1 @ X2.T


def rbf_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)


def polynomial_kernel(x, y, degree=3, coef0=1):
    return (np.dot(x, y) + coef0) ** degree


def sigmoid(z):
    """Sigmoid function with numerical stability"""
    z = np.clip(z, -500, 500)  # Prevent overflow
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


class KernelLogisticRegression:
    def __init__(self, kernel, lambda_=0.01, epochs=1000, batch_size=32,
                 subsample_ratio=0.3, early_stopping_patience=50):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epochs = epochs
        self.batch_size = batch_size
        self.subsample_ratio = subsample_ratio
        self.early_stopping_patience = early_stopping_patience
        self.alphas = None
        self.X_support = None
        self.best_alphas = None
        self.best_loss = float('inf')
        self.patience_counter = 0

    def _compute_kernel_matrix(self, X1, X2=None):
        """Compute kernel matrix efficiently with vectorization where possible"""
        if X2 is None:
            X2 = X1

        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))

        # For RBF kernel, we can vectorize
        if hasattr(self.kernel, 'name') and 'RBF' in self.kernel.name:
            # Extract gamma from kernel name or use default
            gamma = 0.1  # default
            if hasattr(self.kernel, 'name'):
                import re
                match = re.search(r'gamma=([0-9.]+)', self.kernel.name)
                if match:
                    gamma = float(match.group(1))

            # Vectorized RBF computation
            X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
            X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T
            K = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
            K = np.exp(-gamma * K)
        else:
            # Fallback to element-wise computation for other kernels
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.kernel(X1[i], X2[j])

        return K

    def _subsample_support_vectors(self, X):
        """Reduce the number of support vectors to speed up computation"""
        n_samples = X.shape[0]
        n_support = max(int(n_samples * self.subsample_ratio), 50)  # At least 50 points
        n_support = min(n_support, n_samples)  # Don't exceed available samples

        indices = np.random.choice(n_samples, n_support, replace=False)
        return X[indices], indices

    def fit(self, X, y):
        """Train with optimizations: subsampling, mini-batch, early stopping"""
        # Convert labels to {-1, 1} format
        y = np.where(y == 0, -1, y)

        # Subsample support vectors for efficiency
        self.X_support, support_indices = self._subsample_support_vectors(X)
        n_support = self.X_support.shape[0]

        # Initialize alphas
        self.alphas = np.random.normal(0, 0.01, n_support)

        # Precompute kernel matrix for support vectors
        print(f"     🧮 Computing kernel matrix ({n_support}x{n_support})...")
        K_support = self._compute_kernel_matrix(self.X_support)

        n_samples = X.shape[0]

        print(f"     🚀 Training with {n_support} support vectors, batch_size={self.batch_size}")

        for epoch in range(self.epochs):
            # Mini-batch training
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Compute kernel matrix between batch and support vectors
                K_batch = self._compute_kernel_matrix(X_batch, self.X_support)

                # Forward pass
                scores = K_batch @ self.alphas
                probabilities = sigmoid(scores)

                # Convert probabilities to {0,1} for loss computation
                y_batch_01 = np.where(y_batch == -1, 0, 1)
                probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

                # Compute loss
                bce_loss = -np.mean(y_batch_01 * np.log(probabilities) +
                                    (1 - y_batch_01) * np.log(1 - probabilities))
                reg_loss = self.lambda_ * np.sum(self.alphas ** 2)
                batch_loss = bce_loss + reg_loss
                epoch_loss += batch_loss
                n_batches += 1

                # Compute gradients
                errors = probabilities - y_batch_01
                gradients = K_batch.T @ errors / len(y_batch) + 2 * self.lambda_ * self.alphas

                # Adaptive learning rate
                learning_rate = 0.01 / (1 + 0.001 * epoch)

                # Update alphas
                self.alphas -= learning_rate * gradients

            avg_loss = epoch_loss / n_batches

            # Early stopping
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_alphas = self.alphas.copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Print progress less frequently
            if epoch % max(1, self.epochs // 10) == 0:
                print(f"       Epoch {epoch:4d}/{self.epochs}: loss={avg_loss:.6f}")

            if self.patience_counter >= self.early_stopping_patience:
                print(f"       Early stopping at epoch {epoch}")
                break

        # Use best alphas
        if self.best_alphas is not None:
            self.alphas = self.best_alphas

    def predict_proba(self, X):
        """Predict probabilities"""
        if self.alphas is None or self.X_support is None:
            raise ValueError("Model must be fitted before prediction")

        # Compute kernel matrix between X and support vectors
        K = self._compute_kernel_matrix(X, self.X_support)
        scores = K @ self.alphas
        probabilities = sigmoid(scores)

        return probabilities

    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


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


class NamedKernel:
    """Wrapper per dare nomi leggibili ai kernel"""

    def __init__(self, kernel_fn, name):
        self.kernel_fn = kernel_fn
        self.name = name

    def __call__(self, x, y):
        return self.kernel_fn(x, y)

    def __repr__(self):
        return f"Kernel({self.name})"

    def __str__(self):
        return self.name


def create_named_kernels(gamma_values, degree_values, coef0_values):
    """Crea kernel con nomi leggibili"""

    rbf_kernels = []
    for gamma in gamma_values:
        kernel_fn = lambda x, y, g=gamma: rbf_kernel(x, y, gamma=g)
        named_kernel = NamedKernel(kernel_fn, f"RBF(gamma={gamma})")
        rbf_kernels.append(named_kernel)

    poly_kernels = []
    for degree in degree_values:
        for coef0 in coef0_values:
            kernel_fn = lambda x, y, d=degree, c=coef0: polynomial_kernel(x, y, degree=d, coef0=c)
            named_kernel = NamedKernel(kernel_fn, f"Poly(degree={degree}, coef0={coef0})")
            poly_kernels.append(named_kernel)

    return rbf_kernels + poly_kernels