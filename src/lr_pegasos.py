import numpy as np

class LogisticRegressionScratch2:
    def __init__(self, learning_rate=0.1, regularization_strength=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_ = regularization_strength
        self.epochs = epochs
        self.w = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        # Ensure y ∈ {−1, +1}
        y = np.where(y <= 0, -1, 1)

        for t in range(1, self.epochs + 1):
            i = np.random.randint(n_samples)
            x_t, y_t = X[i], y[i]

            eta_t = self.learning_rate
            margin = y_t * np.dot(self.w, x_t)
            sigma_term = self.sigmoid(-margin)

            gradient = sigma_term * y_t * x_t - self.lambda_ * self.w
            self.w += eta_t * gradient

    def predict_proba(self, X):
        z = X @ self.w
        return self.sigmoid(z)

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0.5, 1, -1)
