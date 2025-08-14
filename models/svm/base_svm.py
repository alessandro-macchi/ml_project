from src.metrics import comprehensive_evaluation
from src.grid_search import grid_search
import numpy as np


class SVMClassifierScratch:
    """
    Linear SVM classifier trained with the Pegasos algorithm.
    Hyperparameters:
      - lambda_: regularization strength
      - epochs: number of iterations over the training data
    """
    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_
        self.weights = None
        self.bias = 0

    def fit(self, X, y, max_iter=1000):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for t in range(1, max_iter + 1):
            i = np.random.randint(0, n_samples)
            x_i = X[i]
            y_i = y[i]

            eta = 1.0 / (self.lambda_ * t)
            condition = y_i * (np.dot(self.weights, x_i) + self.bias) < 1

            if condition:
                self.weights = (1 - eta * self.lambda_) * self.weights + eta * y_i * x_i
                self.bias += eta * y_i
            else:
                self.weights = (1 - eta * self.lambda_) * self.weights

    def predict(self, X):
        X = np.array(X)
        scores = np.dot(X, self.weights) + self.bias
        return np.where(scores >= 0, 1, -1)  # Return {-1, +1} labels


def run_svm_experiment(X_train, y_train, X_test, y_test, param_grid):
    print("🔍 Grid search for Linear SVM...")
    best_params, best_score = grid_search(X_train, y_train, SVMClassifierScratch, param_grid)
    print(f"✅ Best SVM params: {best_params}, CV Accuracy: {best_score:.4f}")

    model = SVMClassifierScratch(lambda_=best_params["lambda_"])
    model.fit(X_train, y_train, max_iter=best_params["max_iter"])
    preds = model.predict(X_test)

    results = {'svm_custom': comprehensive_evaluation(y_test, preds, "Linear SVM (Custom)")}

    return results, model