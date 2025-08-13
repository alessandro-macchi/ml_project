import pandas as pd
import numpy as np


def load_and_combine_data(red_path: str, white_path: str) -> pd.DataFrame:
    """Load red and white wine datasets and combine them"""
    red = pd.read_csv(red_path, sep=';')
    white = pd.read_csv(white_path, sep=';')

    red["white_type"] = 0  # red
    white["white_type"] = 1  # white

    data = pd.concat([red, white], ignore_index=True)
    return data


def log_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Apply log1p transformation to selected skewed features"""
    for col in columns:
        df[col] = np.log1p(df[col])
    return df


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

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

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


def preprocess_features(df: pd.DataFrame, apply_smote=False) -> tuple:
    """Preprocess wine dataset features and split into train/test sets."""
    df["quality_binary"] = (df["quality"] >= 6).astype(int)

    # Separate features and labels before transformations
    X = df.drop(columns=["quality", "quality_binary"])
    y = df["quality_binary"].values

    # Train-test split first
    X_train, X_test, y_train, y_test = custom_train_test_split(
        X, y, test_size=0.2, random_state=6, stratify=y
    )

    # Log-transform skewed features AFTER splitting
    skewed_cols = ["residual sugar", "free sulfur dioxide",
                   "total sulfur dioxide", "chlorides", "sulphates"]

    X_train = log_transform(X_train, skewed_cols)
    X_test = log_transform(X_test, skewed_cols)

    # Standardize AFTER log transform
    train_mean = X_train.mean()
    train_std = X_train.std()

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    # Apply SMOTE-like oversampling
    if apply_smote:
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))

        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)

        n_needed = class_counts[majority_class] - class_counts[minority_class]

        X_minority = X_train[y_train == minority_class]

        synthetic_samples = generate_synthetic_samples(X_minority, N=n_needed, k=5)
        synthetic_labels = np.array([minority_class] * len(synthetic_samples))

        X_train = np.vstack([X_train, synthetic_samples])
        y_train = np.concatenate([y_train, synthetic_labels])

    return X_train, X_test.values, y_train, y_test
