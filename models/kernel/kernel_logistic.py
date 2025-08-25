from core import comprehensive_evaluation, grid_search, sigmoid
import numpy as np


class KernelLogisticRegression:
    def __init__(self, kernel, lambda_=0.01, epochs=1000, batch_size=64,
                 subsample_ratio=0.3, early_stopping_patience=20, random_state=None):
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
        self.losses = []

    def _compute_kernel_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1

        if hasattr(X1, 'values'):
            X1 = X1.values
        if hasattr(X2, 'values'):
            X2 = X2.values

        X1 = np.asarray(X1)
        X2 = np.asarray(X2)

        kernel_name = getattr(self.kernel, 'name', str(self.kernel))

        if 'RBF' in kernel_name:
            gamma = 0.1
            import re
            match = re.search(r'gamma=([0-9.]+)', kernel_name)
            if match:
                gamma = float(match.group(1))

            X1_sq_norm = np.sum(X1 ** 2, axis=1, keepdims=True)
            X2_sq_norm = np.sum(X2 ** 2, axis=1, keepdims=True).T

            squared_distances = X1_sq_norm + X2_sq_norm - 2 * np.dot(X1, X2.T)

            squared_distances = np.maximum(squared_distances, 0)

            K = np.exp(-gamma * squared_distances)
            return K

        elif 'Poly' in kernel_name:
            degree = 3
            coef0 = 1
            import re
            deg_match = re.search(r'degree=([0-9]+)', kernel_name)
            coef_match = re.search(r'coef0=([0-9.]+)', kernel_name)
            if deg_match:
                degree = int(deg_match.group(1))
            if coef_match:
                coef0 = float(coef_match.group(1))

            K = np.power(np.dot(X1, X2.T) + coef0, degree)
            return K

        elif 'linear' in kernel_name.lower():
            K = np.dot(X1, X2.T)
            return K

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
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)

        n_samples = X_array.shape[0]
        n_support = max(int(n_samples * self.subsample_ratio), 50)
        n_support = min(n_support, n_samples)

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

        self.X_train_original = X.copy()
        self.y_train_original = y.copy()
        self.losses = []

        X_support_subset, support_indices = self._subsample_support_vectors(X)
        n_support = X_support_subset.shape[0]

        alphas_subset = self.random_state.normal(0, 0.01, n_support)

        print(f"     🧮 Computing kernel matrix ({n_support}x{n_support})...")
        K_support = self._compute_kernel_matrix(X_support_subset)

        n_samples = X.shape[0]

        print(f"     🚀 Training with {n_support} support vectors, batch_size={self.batch_size}")

        for epoch in range(self.epochs):
            indices = self.random_state.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                K_batch = self._compute_kernel_matrix(X_batch, X_support_subset)

                scores = K_batch @ alphas_subset

                margins = -y_batch * scores
                logistic_loss = np.mean(np.logaddexp(0, margins))
                reg_loss = self.lambda_ * np.sum(alphas_subset ** 2)
                batch_loss = logistic_loss + reg_loss

                epoch_loss += batch_loss
                n_batches += 1

                sigmoid_margins = sigmoid(margins)
                gradient_factor = -y_batch * sigmoid_margins
                gradients = K_batch.T @ gradient_factor / len(y_batch) + 2 * self.lambda_ * alphas_subset

                learning_rate = 0.01 / (1 + 0.001 * epoch)

                alphas_subset -= learning_rate * gradients

            avg_loss = epoch_loss / n_batches
            self.losses.append(avg_loss)

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                best_alphas_subset = alphas_subset.copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if epoch % max(1, self.epochs // 10) == 0:
                print(f"       Epoch {epoch:4d}/{self.epochs}: loss={avg_loss:.6f}")

            if self.patience_counter >= self.early_stopping_patience:
                print(f"       Early stopping at epoch {epoch}")
                break

        try:
            if 'best_alphas_subset' in locals():
                trained_alphas_subset = best_alphas_subset
            else:
                trained_alphas_subset = alphas_subset

            self.X_support = self.X_train_original
            n_total = len(self.X_train_original)

            K_all_to_subset = self._compute_kernel_matrix(self.X_train_original, X_support_subset)

            subset_scores = K_all_to_subset @ trained_alphas_subset

            target_scores = self.y_train_original.astype(float)

            K_full = self._compute_kernel_matrix(self.X_train_original)

            K_reg = K_full + self.lambda_ * np.eye(n_total)

            try:
                self.alphas = np.linalg.solve(K_reg, target_scores)
            except np.linalg.LinAlgError:
                self.alphas = np.linalg.pinv(K_reg) @ target_scores

        except Exception as e:
            print(f"       ⚠️ Warning: Could not compute full alphas, using subset approach: {e}")
            self.X_support = X_support_subset
            self.alphas = trained_alphas_subset if 'trained_alphas_subset' in locals() else alphas_subset

    def predict_proba(self, X):
        """Predict probabilities"""
        if self.alphas is None or self.X_support is None:
            raise ValueError("Model must be fitted before prediction")

        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.array(X)

        K = self._compute_kernel_matrix(X, self.X_support)
        scores = K @ self.alphas
        probabilities = sigmoid(scores)

        return probabilities

    def predict(self, X):
        """Predict class labels"""
        if self.alphas is None or self.X_support is None:
            raise ValueError("Model must be fitted before prediction")

        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.array(X)

        K = self._compute_kernel_matrix(X, self.X_support)
        scores = K @ self.alphas
        return np.where(scores >= 0, 1, -1)

    @classmethod
    def run_kernel_logistic_regression_experiment(cls, X_train, y_train, X_test, y_test, param_grid):
        best_params, best_score = grid_search(X_train, y_train, cls, param_grid)

        model = cls(
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