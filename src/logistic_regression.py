import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, regularization_strength=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_ = regularization_strength
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        reg_loss = self.lambda_ * np.sum(self.weights ** 2)
        return bce_loss + reg_loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []
        y = np.where(y == -1, 0, y)

        for epoch in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            dz = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, dz) + (2 * self.lambda_ / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(dz)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.where(probas >= 0.5, 1, 0)