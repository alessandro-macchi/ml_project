import numpy as np
from core.math_utils import euclidean_distance


def custom_train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
    """Basic train-test split with optional stratification"""
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(len(y))

    if stratify is not None:
        classes, y_indices = np.unique(stratify, return_inverse=True)
        test_indices = []

        for cls in range(len(classes)):
            cls_indices = indices[y_indices == cls]
            np.random.shuffle(cls_indices)
            n_test = int(len(cls_indices) * test_size)
            test_indices.extend(cls_indices[:n_test])

        test_indices = np.array(test_indices)
    else:
        np.random.shuffle(indices)
        n_test = int(len(y) * test_size)
        test_indices = indices[:n_test]

    train_indices = np.setdiff1d(indices, test_indices)

    return X.iloc[train_indices], X.iloc[test_indices], y[train_indices], y[test_indices]



def generate_synthetic_samples(X_minority_df, N, k=5):
    X_minority = X_minority_df.values  # convert once for easier math
    synthetic_samples = []

    for _ in range(N):
        i = np.random.randint(0, len(X_minority))
        x_i = X_minority[i]

        # Find k nearest neighbors
        distances = [euclidean_distance(x_i, x_j) for j, x_j in enumerate(X_minority) if j != i]
        neighbors_idx = np.argsort(distances)[:k]
        neighbors = [X_minority[j] for j in neighbors_idx]

        x_nn = neighbors[np.random.randint(0, k)]

        gap = np.random.rand()
        synthetic = x_i + gap * (x_nn - x_i)
        synthetic_samples.append(synthetic)

    return np.array(synthetic_samples)