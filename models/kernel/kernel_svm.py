from core import comprehensive_evaluation, grid_search, sigmoid
import numpy as np

class KernelPegasosSVM:
    def __init__(self, kernel, lambda_=0.01, max_iter=1000, verbose=False, random_state=None,
                 track_losses=False):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = np.random.RandomState(random_state)
        self.track_losses = track_losses
        self.support_vectors = []
        self.support_labels = []
        self.alphas = []
        self.losses = [] if track_losses else None

    def fit(self, X, y, max_iter=None):
        if max_iter is not None:
            self.max_iter = max_iter

        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.array(X)

        if hasattr(y, 'values'):
            y = y.values
        else:
            y = np.array(y)

        if self.track_losses:
            self.X_train = X
            self.y_train = y

        if set(np.unique(y)) != {-1, 1}:
            raise ValueError("Pegasos requires binary classification with labels in {-1, 1}")

        n_samples = X.shape[0]

        self.support_vectors = []
        self.support_labels = []
        self.alphas = []
        if self.track_losses:
            self.losses = []

        if self.verbose:
            mode = "with loss tracking" if self.track_losses else "fast mode"
            print(f"     🚀 Training Kernel Pegasos SVM for {self.max_iter} iterations ({mode})...")

        margin_violations = 0

        if self.track_losses:
            loss_frequency = 50
        else:
            loss_frequency = self.max_iter + 1

        for t in range(1, self.max_iter + 1):
            i = self.random_state.randint(0, n_samples)
            x_t = X[i]
            y_t = y[i]

            if len(self.support_vectors) == 0:
                decision_value = 0.0
            else:
                kernel_values = np.array([
                    self.kernel(sv, x_t)
                    for sv in self.support_vectors
                ])
                decision_value = np.sum(
                    np.array(self.alphas) * np.array(self.support_labels) * kernel_values
                )

            eta_t = 1.0 / (self.lambda_ * t)
            margin = y_t * decision_value

            if margin < 1:
                self.support_vectors.append(x_t.copy())
                self.support_labels.append(y_t)
                self.alphas.append(eta_t)
                margin_violations += 1

            decay_factor = max(0, 1 - eta_t * self.lambda_)
            self.alphas = [alpha * decay_factor for alpha in self.alphas]

            prune_frequency = 500 if self.track_losses else 1000
            if t % prune_frequency == 0:
                self._prune_support_vectors(threshold=1e-8)

            if self.track_losses and t % loss_frequency == 0:
                loss = self._compute_kernel_svm_loss_fast()  # Use fast approximation
                self.losses.append(loss)

            if self.verbose and t % max(1, self.max_iter // 5) == 0:
                n_sv = len(self.support_vectors)
                violation_rate = margin_violations / t
                if self.track_losses and self.losses:
                    current_loss = self.losses[-1]
                    print(
                        f"       Iteration {t:4d}/{self.max_iter}: {n_sv} SVs, {violation_rate:.3f} violation rate, loss: {current_loss:.4f}")
                else:
                    print(f"       Iteration {t:4d}/{self.max_iter}: {n_sv} SVs, {violation_rate:.3f} violation rate")

        if self.verbose:
            print(
                f"     ✅ Training completed. Final: {len(self.support_vectors)} SVs, {margin_violations} total violations")

        self._prune_support_vectors(threshold=1e-10)

    def _compute_kernel_svm_loss_fast(self):
        if len(self.support_vectors) == 0:
            return 1.0

        try:
            n_sv = len(self.support_vectors)
            if n_sv == 0:
                return 1.0

            decision_values = []
            for i, sv in enumerate(self.support_vectors):
                kernel_vals = np.array([
                    self.kernel(other_sv, sv)
                    for other_sv in self.support_vectors
                ])
                decision_val = np.sum(
                    np.array(self.alphas) * np.array(self.support_labels) * kernel_vals
                )
                decision_values.append(decision_val)

            decision_values = np.array(decision_values)

            margins = np.array(self.support_labels) * decision_values
            hinge_losses = np.maximum(0, 1 - margins)
            avg_hinge_loss = np.mean(hinge_losses)

            reg_loss = (self.lambda_ / 2) * np.sum(np.array(self.alphas) ** 2)

            sv_ratio = len(self.support_vectors) / len(self.X_train) if hasattr(self, 'X_train') else 1.0
            estimated_loss = avg_hinge_loss * sv_ratio + reg_loss

            return estimated_loss

        except Exception:
            sv_ratio = len(self.support_vectors) / len(self.X_train) if hasattr(self, 'X_train') else 0.1
            alpha_magnitude = np.mean(np.abs(self.alphas)) if self.alphas else 0.1
            return sv_ratio * alpha_magnitude

    def _prune_support_vectors(self, threshold=1e-8):
        if not self.alphas:
            return

        significant_indices = [
            i for i, alpha in enumerate(self.alphas)
            if abs(alpha) > threshold
        ]

        if significant_indices:
            self.support_vectors = [self.support_vectors[i] for i in significant_indices]
            self.support_labels = [self.support_labels[i] for i in significant_indices]
            self.alphas = [self.alphas[i] for i in significant_indices]
        elif self.support_vectors:
            max_idx = np.argmax([abs(alpha) for alpha in self.alphas])
            self.support_vectors = [self.support_vectors[max_idx]]
            self.support_labels = [self.support_labels[max_idx]]
            self.alphas = [self.alphas[max_idx]]

    def decision_function(self, X):
        if not self.support_vectors:
            return np.full(X.shape[0], -0.1)

        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        decision_values = []

        for x in X:
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
        Predict class labels.

        Args:
            X: Input samples (n_samples, n_features)

        Returns:
            Predicted labels {-1, +1}
        """
        decision_values = self.decision_function(X)
        return np.where(decision_values >= 0, 1, -1)  # Return {-1, +1} labels

    def predict_proba(self, X):
        decision_values = self.decision_function(X)

        probabilities = sigmoid(decision_values)

        return probabilities

    @classmethod
    def run_kernel_svm_experiment(cls, X_train, y_train, X_test, y_test, param_grid):
        """
        Two-phase experiment:
        1. Grid search with ALL parameters from param_grid (fast, no loss tracking)
        2. Train final model with best parameters (with loss tracking for plots)
        """

        best_params, best_score = grid_search(
            X_train, y_train,
            lambda **kwargs: cls(track_losses=False, verbose=False, **kwargs),  # Fast mode
            param_grid  # Use your COMPLETE param_grid as specified
        )

        final_model = cls(
            kernel=best_params["kernel"],
            lambda_=best_params["lambda_"],
            max_iter=best_params["max_iter"],
            track_losses=True,
            verbose=True,
            random_state=12
        )

        final_model.fit(X_train, y_train)
        preds = final_model.predict(X_test)

        print(f"📊 Number of support vectors: {len(final_model.support_vectors)}")
        print(f"📈 Loss tracking: {len(final_model.losses)} loss points recorded")

        results = {'ksvm_custom': comprehensive_evaluation(y_test, preds, "Kernel SVM (Pegasos)")}

        return results, final_model