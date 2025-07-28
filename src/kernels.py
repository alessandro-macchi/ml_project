import numpy as np

def linear_kernel(X1, X2):
    return X1 @ X2.T

def rbf_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

def polynomial_kernel(x, y, degree=3, coef0=1):
    return (np.dot(x, y) + coef0) ** degree

class KernelPegasosSVM:
    def __init__(self, kernel_fn, lambda_=0.01, max_iter=1000):
        self.kernel_fn = kernel_fn
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.support_vectors = []
        self.support_labels = []
        self.alphas = []  # alpha_i = y_i / (λ t)
        self.coeff_decay = []  # store (1 - 1/t) coefficients

    def fit(self, X, y):
        n_samples = X.shape[0]

        for t in range(1, self.max_iter + 1):
            i = np.random.randint(0, n_samples)
            x_t = X[i]
            y_t = y[i]

            # Compute g_t(x_t)
            if len(self.support_vectors) == 0:
                g_x_t = 0
            else:
                g_x_t = sum(
                    alpha * self.kernel_fn(sv, x_t)
                    for alpha, sv in zip(self.alphas, self.support_vectors)
                )

            # Compute margin loss
            h_t = 1 - y_t * g_x_t

            # Decay old alphas: scale them by (1 - 1/t)
            self.alphas = [(1 - 1 / t) * alpha for alpha in self.alphas]

            # If the hinge loss is positive, perform update
            if h_t > 0:
                alpha_t = (y_t / (self.lambda_ * t))
                self.support_vectors.append(x_t)
                self.alphas.append(alpha_t)

    def predict(self, X):
        preds = []
        for x in X:
            result = sum(
                alpha * self.kernel_fn(sv, x)
                for alpha, sv in zip(self.alphas, self.support_vectors)
            )
            preds.append(np.sign(result))
        return np.array(preds)


class NamedKernel:
    """Wrapper per dare nomi leggibili ai kernel"""

    def __init__(self, kernel_fn, name):
        self.kernel_fn = kernel_fn
        self.name = name

    def __call__(self, x, y):
        return self.kernel_fn(x, y)

    def __repr__(self):
        return f"Kernel({self.name})"

    def __str__(self):
        return self.name


def create_named_kernels(gamma_values, degree_values, coef0_values):
    """Crea kernel con nomi leggibili"""

    rbf_kernels = []
    for gamma in gamma_values:
        kernel_fn = lambda x, y, g=gamma: rbf_kernel(x, y, gamma=g)
        named_kernel = NamedKernel(kernel_fn, f"RBF(gamma={gamma})")
        rbf_kernels.append(named_kernel)

    poly_kernels = []
    for degree in degree_values:
        for coef0 in coef0_values:
            kernel_fn = lambda x, y, d=degree, c=coef0: polynomial_kernel(x, y, degree=d, coef0=c)
            named_kernel = NamedKernel(kernel_fn, f"Poly(degree={degree}, coef0={coef0})")
            poly_kernels.append(named_kernel)

    return rbf_kernels + poly_kernels
