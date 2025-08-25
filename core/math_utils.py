import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))