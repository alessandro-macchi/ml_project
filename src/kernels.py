import numpy as np

def linear_kernel(X1, X2):
    return X1 @ X2.T

def polynomial_kernel(X1, X2, degree=3, coef0=1):
    return (X1 @ X2.T + coef0) ** degree

def rbf_kernel(X1, X2, gamma=0.1):
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    sq_dists = X1_sq + X2_sq - 2 * X1 @ X2.T
    return np.exp(-gamma * sq_dists)
