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
        """Sigmoid function with numerical stability"""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def compute_loss(self, y_true, y_pred):
        """Binary cross entropy loss with regularization"""
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Prevent log(0)

        # Binary cross entropy
        bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # L2 regularization
        reg_loss = self.lambda_ * np.sum(self.weights ** 2)

        return bce_loss + reg_loss

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        # Ensure y is in {0, 1} format (not {-1, 1})
        y = np.where(y == -1, 0, y)  # Convert -1 to 0 if present

        # Choose between batch GD or stochastic GD
        use_batch = True  # Set to False for SGD like your original

        if use_batch:
            # Batch Gradient Descent (like the target class)
            for epoch in range(self.epochs):
                # Forward pass
                z = np.dot(X, self.weights) + self.bias
                y_pred = self.sigmoid(z)

                # Compute loss
                loss = self.compute_loss(y, y_pred)
                self.losses.append(loss)

                # Compute gradients
                dz = y_pred - y
                dw = (1 / n_samples) * np.dot(X.T, dz) + (2 * self.lambda_ / n_samples) * self.weights
                db = (1 / n_samples) * np.sum(dz)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        else:
            # Stochastic Gradient Descent (your original approach, but corrected)
            for t in range(1, self.epochs + 1):
                i = np.random.randint(n_samples)
                x_i, y_i = X[i], y[i]

                # Forward pass for single sample
                z_i = np.dot(self.weights, x_i) + self.bias
                y_pred_i = self.sigmoid(z_i)

                # Compute gradients for single sample
                error = y_pred_i - y_i
                dw = error * x_i + 2 * self.lambda_ * self.weights
                db = error

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                # Compute full loss periodically (for monitoring)
                if t % 100 == 0:
                    z_full = np.dot(X, self.weights) + self.bias
                    y_pred_full = self.sigmoid(z_full)
                    loss = self.compute_loss(y, y_pred_full)
                    self.losses.append(loss)

    def predict_proba(self, X):
        """Predict class probabilities"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        """Predict class labels - returns {0, 1} to match your data format"""
        probas = self.predict_proba(X)
        return np.where(probas >= 0.5, 1, 0)