from src.metrics import comprehensive_evaluation
from src.grid_search import grid_search
import numpy as np


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, regularization_strength=0.01, epochs=1000, random_state=None):
        self.learning_rate = learning_rate
        self.lambda_ = regularization_strength
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []
        self.random_state = np.random.RandomState(random_state)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def compute_loss(self, y_true, scores):
        """Margin-based logistic loss for {-1, +1} labels"""
        # Logistic loss: log(1 + exp(-y * score))
        margins = -y_true * scores  # Negative margin
        logistic_loss = np.mean(np.logaddexp(0, margins))
        reg_loss = self.lambda_ * np.sum(self.weights ** 2)
        return logistic_loss + reg_loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for epoch in range(self.epochs):
            scores = np.dot(X, self.weights) + self.bias
            loss = self.compute_loss(y, scores)
            self.losses.append(loss)

            # ∂L/∂w = -X^T * (y * sigmoid(-y * scores)) + 2λw
            margins = -y * scores
            sigmoid_margins = self.sigmoid(margins)
            gradient_factor = -y * sigmoid_margins

            dw = (1 / n_samples) * np.dot(X.T, gradient_factor) + (2 * self.lambda_ / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(gradient_factor)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return self.sigmoid(scores)

    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return np.where(scores >= 0, 1, -1)


def run_logistic_regression_experiment(X_train, y_train, X_test, y_test, param_grid):
    print("🔍 Grid search for Logistic Regression...")
    best_params, best_score = grid_search(X_train, y_train, LogisticRegressionScratch, param_grid)
    print(f"✅ Best LR params: {best_params}, CV Accuracy: {best_score:.4f}")

    model = LogisticRegressionScratch(random_state=24, **best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results = {'lr_custom': comprehensive_evaluation(y_test, preds, "Logistic Regression (Custom)")}

    return results, model