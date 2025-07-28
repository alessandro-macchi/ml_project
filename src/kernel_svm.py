import numpy as np

class KernelSVM:
    def __init__(self, kernel_fn, lambda_=0.01, max_iter=1000):
        self.kernel_fn = kernel_fn
        self.lambda_ = lambda_
        self.max_iter = max_iter
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
