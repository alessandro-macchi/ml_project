from core import comprehensive_evaluation, grid_search, sigmoid
import numpy as np


# =============================================================================
# EFFICIENT KERNEL SVM APPROACH: Two-Phase Training
# =============================================================================

class KernelPegasosSVM:
    """
    Kernel SVM using the Pegasos algorithm.
    OPTIMIZED: Fast grid search + detailed loss tracking for best params only
    """

    def __init__(self, kernel, lambda_=0.01, max_iter=1000, verbose=False, random_state=None,
                 track_losses=False):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = np.random.RandomState(random_state)
        self.track_losses = track_losses  # NEW: Only track when needed

        # Support vectors and their coefficients
        self.support_vectors = []
        self.support_labels = []
        self.alphas = []
        self.losses = [] if track_losses else None

    def fit(self, X, y, max_iter=None):
        """
        OPTIMIZED Training: Fast mode for grid search, detailed mode for final training
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

        # Store training data only if tracking losses
        if self.track_losses:
            self.X_train = X
            self.y_train = y

        if set(np.unique(y)) != {-1, 1}:
            raise ValueError("Pegasos requires binary classification with labels in {-1, 1}")

        n_samples = X.shape[0]

        # Clear previous training data
        self.support_vectors = []
        self.support_labels = []
        self.alphas = []
        if self.track_losses:
            self.losses = []

        if self.verbose:
            mode = "with loss tracking" if self.track_losses else "fast mode"
            print(f"     🚀 Training Kernel Pegasos SVM for {self.max_iter} iterations ({mode})...")

        margin_violations = 0

        # OPTIMIZATION: Different loss computation frequencies
        if self.track_losses:
            loss_frequency = 50  # Compute every 50 iterations for detailed tracking
        else:
            loss_frequency = self.max_iter + 1  # Never compute (skip)

        for t in range(1, self.max_iter + 1):
            # Sample a random training example
            i = self.random_state.randint(0, n_samples)
            x_t = X[i]
            y_t = y[i]

            # Compute decision function value
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

            # Decay existing coefficients
            decay_factor = max(0, 1 - eta_t * self.lambda_)
            self.alphas = [alpha * decay_factor for alpha in self.alphas]

            # OPTIMIZATION: Less frequent pruning in fast mode
            prune_frequency = 500 if self.track_losses else 1000
            if t % prune_frequency == 0:
                self._prune_support_vectors(threshold=1e-8)

            # CONDITIONAL LOSS COMPUTATION: Only when tracking losses
            if self.track_losses and t % loss_frequency == 0:
                loss = self._compute_kernel_svm_loss_fast()  # Use fast approximation
                self.losses.append(loss)

            # OPTIMIZATION: Less frequent progress reporting
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

        # Final pruning
        self._prune_support_vectors(threshold=1e-10)

    def _compute_kernel_svm_loss_fast(self):
        """
        FAST loss approximation using support vectors only (not full dataset)
        This is much faster than computing on all training examples
        """
        if len(self.support_vectors) == 0:
            return 1.0

        try:
            # OPTIMIZATION: Estimate loss using support vectors only
            # This gives a good approximation of the true loss much faster

            n_sv = len(self.support_vectors)
            if n_sv == 0:
                return 1.0

            # Quick hinge loss approximation based on support vector margins
            decision_values = []
            for i, sv in enumerate(self.support_vectors):
                # Self-kernel evaluation (faster than full computation)
                kernel_vals = np.array([
                    self.kernel(other_sv, sv)
                    for other_sv in self.support_vectors
                ])
                decision_val = np.sum(
                    np.array(self.alphas) * np.array(self.support_labels) * kernel_vals
                )
                decision_values.append(decision_val)

            decision_values = np.array(decision_values)

            # Hinge loss on support vectors (these are the "difficult" examples)
            margins = np.array(self.support_labels) * decision_values
            hinge_losses = np.maximum(0, 1 - margins)
            avg_hinge_loss = np.mean(hinge_losses)

            # Regularization term
            reg_loss = (self.lambda_ / 2) * np.sum(np.array(self.alphas) ** 2)

            # Scale by support vector ratio to approximate full loss
            sv_ratio = len(self.support_vectors) / len(self.X_train) if hasattr(self, 'X_train') else 1.0
            estimated_loss = avg_hinge_loss * sv_ratio + reg_loss

            return estimated_loss

        except Exception:
            # Ultra-fast fallback: loss based on support vector statistics
            sv_ratio = len(self.support_vectors) / len(self.X_train) if hasattr(self, 'X_train') else 0.1
            alpha_magnitude = np.mean(np.abs(self.alphas)) if self.alphas else 0.1
            return sv_ratio * alpha_magnitude

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
        Predict class labels.

        Args:
            X: Input samples (n_samples, n_features)

        Returns:
            Predicted labels {-1, +1}
        """
        decision_values = self.decision_function(X)
        return np.where(decision_values >= 0, 1, -1)  # Return {-1, +1} labels

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

    @classmethod
    def run_kernel_svm_experiment(cls, X_train, y_train, X_test, y_test, param_grid):
        """
        Two-phase experiment:
        1. Grid search with ALL parameters from param_grid (fast, no loss tracking)
        2. Train final model with best parameters (with loss tracking for plots)
        """
        print("     📊 Phase 1: Grid search across all parameter combinations...")

        # PHASE 1: COMPLETE GRID SEARCH with all your parameters
        # Use fast mode (no loss tracking) to test all combinations quickly
        best_params, best_score = grid_search(
            X_train, y_train,
            lambda **kwargs: cls(track_losses=False, verbose=False, **kwargs),  # Fast mode
            param_grid  # Use your COMPLETE param_grid as specified
        )

        print(f"     📊 Phase 2: Training final model with best parameters and loss tracking...")

        # PHASE 2: Train final model with best parameters AND loss tracking
        final_model = cls(
            kernel=best_params["kernel"],
            lambda_=best_params["lambda_"],
            max_iter=best_params["max_iter"],
            track_losses=True,  # Enable loss tracking for plots
            verbose=True,  # Enable verbose for final training
            random_state=12  # Fixed seed for reproducibility
        )

        # Train final model with loss tracking enabled
        final_model.fit(X_train, y_train)

        # Make predictions
        preds = final_model.predict(X_test)

        print(f"📊 Number of support vectors: {len(final_model.support_vectors)}")
        print(f"📈 Loss tracking: {len(final_model.losses)} loss points recorded")

        # Comprehensive evaluation
        results = {'ksvm_custom': comprehensive_evaluation(y_test, preds, "Kernel SVM (Pegasos)")}

        return results, final_model