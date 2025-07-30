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
    def __init__(self, kernel_fn, lambda_=0.01, max_iter=1000):
        self.kernel_fn = kernel_fn
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.support_vectors = []
        self.support_labels = []
        self.alphas = []  # alpha_i = y_i / (λ t)
        self.coeff_decay = []  # store (1 - 1/t) coefficients

    def fit(self, X, y, max_iter=None):
        if max_iter is not None:
            self.max_iter = max_iter

        # Ensure labels are -1 or +1
        y = np.where(y == 0, -1, y)
        if set(np.unique(y)) != {-1, 1}:
            raise ValueError("Pegasos requires binary classification with labels in {-1, 1}")

        n_samples = X.shape[0]

        for t in range(1, self.max_iter + 1):
            i = np.random.randint(0, n_samples)
            x_t = X[i]
            y_t = y[i]

            if len(self.support_vectors) == 0:
                g_x_t = 0
            else:
                g_x_t = sum(
                    alpha * y_i * self.kernel_fn(sv, x_t)
                    for alpha, y_i, sv in zip(self.alphas, self.support_labels, self.support_vectors)
                )

            # Margin violation
            if y_t * g_x_t < 1:
                alpha_t = 1 / (self.lambda_ * t)
                self.support_vectors.append(x_t)
                self.support_labels.append(y_t)
                self.alphas.append(alpha_t)

            # Decay all previous alphas
            self.alphas = [(1 - 1 / t) * alpha for alpha in self.alphas]

    def predict(self, X):
        preds = []
        for x in X:
            result = sum(
                alpha * y_i * self.kernel_fn(sv, x)
                for alpha, y_i, sv in zip(self.alphas, self.support_labels, self.support_vectors)
            )
            preds.append(np.sign(result))
        return np.array(preds)


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