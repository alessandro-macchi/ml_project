import numpy as np


def linear_kernel(X1, X2):
    return X1 @ X2.T

def rbf_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)


def polynomial_kernel(x, y, degree=3, coef0=1):
    return (np.dot(x, y) + coef0) ** degree


def sigmoid(z):
    """Sigmoid function with numerical stability"""
    z = np.clip(z, -500, 500)  # Prevent overflow
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

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

def get_model_names():
    """
    Get display names for all models

    Returns:
        dict: Mapping of model keys to display names
    """
    return {
        'lr_custom': 'Logistic Regression',
        'svm_custom': 'Linear SVM',
        'klr_custom': 'Kernel Logistic Regression',
        'ksvm_custom': 'Kernel SVM'
    }

def get_wine_feature_names():
    """Get wine feature names for analysis"""
    return [
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'wine_type'
    ]