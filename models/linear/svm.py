from core import comprehensive_evaluation, grid_search
import numpy as np


class SVMClassifierScratch:
    """
    Linear SVM classifier trained with the Pegasos algorithm.
    Hyperparameters:
      - lambda_: regularization strength
      - epochs: number of iterations over the training data
      - random_state: for reproducible random sampling
    """

    def __init__(self, lambda_=0.01, random_state=None):
        self.lambda_ = lambda_
        self.weights = None
        self.bias = 0
        self.random_state = np.random.RandomState(random_state)
        self.losses = []  # ADD THIS LINE to track losses

    def fit(self, X, y, max_iter=1000):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.weights = self.random_state.normal(0, 0.01, n_features)
        self.bias = 0
        self.losses = []  # RESET losses for new training

        for t in range(1, max_iter + 1):
            # Sample a random training example
            i = self.random_state.randint(0, n_samples)
            x_i = X[i]
            y_i = y[i]

            eta = 1.0 / (self.lambda_ * t)
            condition = y_i * (np.dot(self.weights, x_i) + self.bias) < 1

            if condition:
                self.weights = (1 - eta * self.lambda_) * self.weights + eta * y_i * x_i
                self.bias += eta * y_i
            else:
                self.weights = (1 - eta * self.lambda_) * self.weights

            # ADD LOSS COMPUTATION: Compute SVM hinge loss every few iterations
            if t % 50 == 0 or t == max_iter:  # Every 50 iterations to avoid slowdown
                # Compute hinge loss: max(0, 1 - y_i * (w^T x_i + b))
                scores = np.dot(X, self.weights) + self.bias
                margins = y * scores
                hinge_losses = np.maximum(0, 1 - margins)
                avg_hinge_loss = np.mean(hinge_losses)

                # Add regularization term: λ/2 * ||w||²
                reg_loss = (self.lambda_ / 2) * np.sum(self.weights ** 2)
                total_loss = avg_hinge_loss + reg_loss

                self.losses.append(total_loss)

    def predict(self, X):
        X = np.array(X)
        scores = np.dot(X, self.weights) + self.bias
        return np.where(scores >= 0, 1, -1)  # Return {-1, +1} labels

    @classmethod
    def run_svm_experiment(cls, X_train, y_train, X_test, y_test, param_grid):
        best_params, best_score = grid_search(X_train, y_train, cls, param_grid)

        # ✅ Use unique random state for this model
        model = cls(lambda_=best_params["lambda_"], random_state=15)
        model.fit(X_train, y_train, max_iter=best_params["max_iter"])
        preds = model.predict(X_test)

        results = {'svm_custom': comprehensive_evaluation(y_test, preds, "Linear SVM (Custom)")}

        return results, model