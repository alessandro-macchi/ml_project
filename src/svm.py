import numpy as np

class SVMClassifierScratch:
    def __init__(self, lambda_=0.01, max_iter=1000):
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.w = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for t in range(1, self.max_iter + 1):
            i = np.random.randint(0, n_samples)
            x_i, y_i = X[i], y[i]

            eta = 1 / (self.lambda_ * t)
            condition = y_i * np.dot(self.w, x_i) < 1

            if condition:
                self.w = (1 - eta * self.lambda_) * self.w + eta * y_i * x_i
            else:
                self.w = (1 - eta * self.lambda_) * self.w

    def predict(self, X):
        return np.sign(np.dot(X, self.w))
