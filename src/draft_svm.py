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
        """
        Train the linear SVM using Pegasos update.
        X: np.ndarray of shape (n_samples, n_features)
        y: binary labels {0,1} converted internally to {-1,1}
        max_iter: total number of SGD updates
        """
        n_samples, n_features = X.shape
        # Initialize weight vector
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert labels to -1 and +1
        y_transformed = np.where(y == 1, 1, -1)

        for t in range(1, max_iter + 1):
            # Sample a random index
            i = np.random.randint(0, n_samples)
            x_i = X[i]
            y_i = y_transformed[i]

            # Learning rate schedule
            eta = 1.0 / (self.lambda_ * t)

            # Check hinge loss condition
            condition = y_i * (np.dot(self.weights, x_i) + self.bias) < 1
            if condition:
                # Update with misclassification
                self.weights = (1 - eta * self.lambda_) * self.weights + eta * y_i * x_i
                self.bias += eta * y_i
            else:
                # Only regularization update
                self.weights = (1 - eta * self.lambda_) * self.weights

    def predict(self, X):
        """
        Predict labels for X. Returns array of {0,1}.
        """
        scores = np.dot(X, self.weights) + self.bias
        preds = np.where(scores >= 0, 1, 0)

        return preds