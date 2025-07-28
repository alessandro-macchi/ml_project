import numpy as np

def linear_kernel(X1, X2):
    return X1 @ X2.T

def rbf_kernel(gamma=0.1):
    return lambda x, y: np.exp(-gamma * np.linalg.norm(x - y) ** 2)

def polynomial_kernel(degree=3, coef0=1):
    return lambda x, y: (np.dot(x, y) + coef0) ** degree

class KernelSVM:
    def __init__(self, kernel_fn, lambda_=0.01, max_iter=1000, gamma=0.1):
        self.kernel_fn = kernel_fn
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.gamma = gamma
        self.support_vectors = []
        self.support_labels = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        for t in range(1, self.max_iter + 1):
            i = np.random.randint(0, n_samples)
            x_i, y_i = X[i], y[i]

            if len(self.support_vectors) == 0:
                f_x_i = 0
            else:
                f_x_i = sum(
                    alpha * y_sv * self.kernel_fn(x_sv, x_i)
                    for alpha, x_sv, y_sv in zip(self.alphas, self.support_vectors, self.support_labels)
                )

            margin = y_i * f_x_i

            if margin < 1:
                alpha_t = 1 / (self.lambda_ * t)
                self.support_vectors.append(x_i)
                self.support_labels.append(y_i)
                self.alphas.append(alpha_t)

    def predict(self, X):
        preds = []
        for x in X:
            result = sum(
                alpha * y_sv * self.kernel_fn(x_sv, x)
                for alpha, x_sv, y_sv in zip(self.alphas, self.support_vectors, self.support_labels)
            )
            preds.append(np.sign(result))
        return np.array(preds)

class KernelLogisticRegression:
    def __init__(self, kernel_fn, lr=0.1, reg=0.01, epochs=100):
        self.kernel_fn = kernel_fn
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.X_train = X
        self.y = y
        self.K = self.kernel_fn(X, X)
        self.alpha = np.zeros(X.shape[0])

        for _ in range(self.epochs):
            preds = self.sigmoid(self.K @ self.alpha)
            gradient = self.K.T @ (preds - y) + self.reg * self.alpha
            self.alpha -= self.lr * gradient

    def predict(self, X_test):
        K_test = self.kernel_fn(X_test, self.X_train)
        preds = self.sigmoid(K_test @ self.alpha)
        return (preds >= 0.5).astype(int)
