from src.metrics import comprehensive_evaluation
from src.grid_search import grid_search
import numpy as np
from src.utils import sigmoid


class KernelLogisticRegression:
    def __init__(self, kernel, lambda_=0.01, epochs=1000, batch_size=32,
                 subsample_ratio=0.3, early_stopping_patience=50, random_state=None):
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
        self.random_state = np.random.RandomState(random_state)

    def _compute_kernel_matrix(self, X1, X2=None):
        """Highly optimized kernel matrix computation - NO nested loops!"""
        if X2 is None:
            X2 = X1

        # Convert to numpy arrays if needed
        if hasattr(X1, 'values'):
            X1 = X1.values
        if hasattr(X2, 'values'):
            X2 = X2.values

        X1 = np.asarray(X1)
        X2 = np.asarray(X2)

        # Get kernel type from name
        kernel_name = getattr(self.kernel, 'name', str(self.kernel))

        # RBF KERNEL - FULLY VECTORIZED (100x faster!)
        if 'RBF' in kernel_name:
            # Extract gamma parameter
            gamma = 0.1  # default
            import re
            match = re.search(r'gamma=([0-9.]+)', kernel_name)
            if match:
                gamma = float(match.group(1))

            # Vectorized RBF computation using broadcasting
            # ||x - y||² = ||x||² + ||y||² - 2⟨x,y⟩
            X1_sq_norm = np.sum(X1 ** 2, axis=1, keepdims=True)  # (n1, 1)
            X2_sq_norm = np.sum(X2 ** 2, axis=1, keepdims=True).T  # (1, n2)

            # Broadcasting: (n1,1) + (1,n2) - (n1,n2) = (n1,n2)
            squared_distances = X1_sq_norm + X2_sq_norm - 2 * np.dot(X1, X2.T)

            # Ensure non-negative (numerical stability)
            squared_distances = np.maximum(squared_distances, 0)

            K = np.exp(-gamma * squared_distances)
            return K

        # POLYNOMIAL KERNEL - FULLY VECTORIZED
        elif 'Poly' in kernel_name:
            # Extract parameters
            degree = 3
            coef0 = 1
            import re
            deg_match = re.search(r'degree=([0-9]+)', kernel_name)
            coef_match = re.search(r'coef0=([0-9.]+)', kernel_name)
            if deg_match:
                degree = int(deg_match.group(1))
            if coef_match:
                coef0 = float(coef_match.group(1))

            # Vectorized: (X1 @ X2.T + coef0)^degree
            K = np.power(np.dot(X1, X2.T) + coef0, degree)
            return K

        # LINEAR KERNEL - SIMPLEST VECTORIZATION
        elif 'linear' in kernel_name.lower():
            K = np.dot(X1, X2.T)
            return K

        # FALLBACK: If custom kernel function, still use nested loops
        # (But this should rarely be hit with your current kernels)
        else:
            print(f"Warning: Using slow fallback for kernel: {kernel_name}")
            n1, n2 = X1.shape[0], X2.shape[0]
            K = np.zeros((n1, n2))
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.kernel(X1[i], X2[j])
            return K

    def _subsample_support_vectors(self, X):
        """Reduce the number of support vectors to speed up computation"""
        # Convert to numpy array if it's a DataFrame
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)

        n_samples = X_array.shape[0]
        n_support = max(int(n_samples * self.subsample_ratio), 50)  # At least 50 points
        n_support = min(n_support, n_samples)  # Don't exceed available samples

        indices = self.random_state.choice(n_samples, n_support, replace=False)
        return X_array[indices], indices

    def fit(self, X, y):
        """Train with optimizations: subsampling, mini-batch, early stopping"""
        # Convert inputs to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.array(X)

        if hasattr(y, 'values'):
            y = y.values
        else:
            y = np.array(y)

        # Subsample support vectors for efficiency
        self.X_support, support_indices = self._subsample_support_vectors(X)
        n_support = self.X_support.shape[0]

        # Initialize alphas
        self.alphas = self.random_state.normal(0, 0.01, n_support)

        # Precompute kernel matrix for support vectors
        print(f"     🧮 Computing kernel matrix ({n_support}x{n_support})...")
        K_support = self._compute_kernel_matrix(self.X_support)

        n_samples = X.shape[0]

        print(f"     🚀 Training with {n_support} support vectors, batch_size={self.batch_size}")

        for epoch in range(self.epochs):
            # Mini-batch training
            indices = self.random_state.permutation(n_samples)
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

                # Margin-based logistic loss for {-1, +1} labels
                margins = -y_batch * scores
                logistic_loss = np.mean(np.logaddexp(0, margins))
                reg_loss = self.lambda_ * np.sum(self.alphas ** 2)
                batch_loss = logistic_loss + reg_loss

                epoch_loss += batch_loss
                n_batches += 1

                # Compute gradients for margin-based loss
                sigmoid_margins = sigmoid(margins)
                gradient_factor = -y_batch * sigmoid_margins
                gradients = K_batch.T @ gradient_factor / len(y_batch) + 2 * self.lambda_ * self.alphas

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

        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.array(X)

        # Compute kernel matrix between X and support vectors
        K = self._compute_kernel_matrix(X, self.X_support)
        scores = K @ self.alphas
        probabilities = sigmoid(scores)

        return probabilities

    def predict(self, X):
        """Predict class labels"""
        if self.alphas is None or self.X_support is None:
            raise ValueError("Model must be fitted before prediction")

        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.array(X)

        # Compute kernel matrix between X and support vectors
        K = self._compute_kernel_matrix(X, self.X_support)
        scores = K @ self.alphas
        return np.where(scores >= 0, 1, -1)


def run_kernel_logistic_regression_experiment(X_train, y_train, X_test, y_test, param_grid):
    print("🔍 Grid search for Kernel Logistic Regression...")
    best_params, best_score = grid_search(X_train, y_train, KernelLogisticRegression, param_grid)
    print(f"✅ Best KLR params: {best_params}, CV Accuracy: {best_score:.4f}")

    model = KernelLogisticRegression(
        kernel=best_params["kernel"],
        lambda_=best_params["lambda_"],
        epochs=best_params["epochs"],
        subsample_ratio=0.3,
        batch_size=64,
        early_stopping_patience=20,
        random_state=6
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results = {'klr_custom': comprehensive_evaluation(y_test, preds, "Kernel Logistic Regression (Custom)")}

    return results, model