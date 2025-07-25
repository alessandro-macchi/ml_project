import numpy as np

class SVMClassifierScratch:
    def __init__(self, learning_rate, regularization_strength):
        self.learning_rate = learning_rate
        self.lambda_ = regularization_strength
        self.weights = None
        self.bias = 0
        self.losses = []

    def fit(self, X, y, epochs=1000):
        # Convert labels to {-1, 1} for hinge loss
        y = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(epochs):
            margins = y * (np.dot(X, self.weights) + self.bias)
            mask = margins < 1
            dw = self.lambda_ * self.weights - np.dot(X[mask].T, y[mask]) / n_samples
            db = -np.mean(y[mask])

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def _compute_loss(self, X, y):
        margins = 1 - y * (np.dot(X, self.weights) + self.bias)
        hinge_loss = np.maximum(0, margins)
        return np.mean(hinge_loss) + (self.lambda_ / 2) * np.sum(self.weights ** 2)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return [1 if i >= 0 else 0 for i in linear_output]
