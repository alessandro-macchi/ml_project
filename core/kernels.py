import numpy as np


def rbf_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

def polynomial_kernel(x, y, degree=3, coef0=1):
    return (np.dot(x, y) + coef0) ** degree

class NamedKernel:
    """Wrapper per dare nomi leggibili ai kernel"""

    def __init__(self, kernel, name):
        self.kernel = kernel
        self.name = name

    def __call__(self, x, y):
        return self.kernel(x, y)

    def __str__(self):
        return self.name

    @staticmethod
    def prepare_param_grid(param_grid):
        """Convert kernel parameters to named kernels for grid search"""
        param_grid = param_grid.copy()

        gamma_values = param_grid.pop('gamma_values', None)
        degree_values = param_grid.pop('degree_values', None)
        coef0_values = param_grid.pop('coef0_values', None)

        if any([gamma_values, degree_values, coef0_values]):
            kernels = create_named_kernels(gamma_values or [], degree_values or [], coef0_values or [1])
            param_grid['kernel'] = kernels

        return param_grid


def create_named_kernels(gamma_values, degree_values, coef0_values):
    """Create named kernels for grid search"""
    kernels = []

    for gamma in gamma_values:
        kernel = lambda x, y, g=gamma: rbf_kernel(x, y, gamma=g)
        kernels.append(NamedKernel(kernel, f"RBF(gamma={gamma})"))

    for degree in degree_values:
        for coef0 in coef0_values:
            kernel = lambda x, y, d=degree, c=coef0: polynomial_kernel(x, y, degree=d, coef0=c)
            kernels.append(NamedKernel(kernel, f"Poly(degree={degree}, coef0={coef0})"))

    return kernels