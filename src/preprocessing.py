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


def preprocess_features(df: pd.DataFrame) -> tuple:
    """Preprocess wine dataset features and split into train/test sets."""
    df["quality_binary"] = (df["quality"] >= 6).astype(int)

    # Log-transform skewed features BEFORE splitting (safe)
    skewed_cols = ["residual sugar", "free sulfur dioxide", "total sulfur dioxide", "chlorides", "sulphates"]
    df = log_transform(df, skewed_cols)

    # Separate features and labels
    X = df.drop(columns=["quality", "quality_binary"])
    y = df["quality_binary"].values

    # Train-test split
    X_train, X_test, y_train, y_test = custom_train_test_split(
        X, y, test_size=0.2, random_state=6, stratify=y
    )

    # Standardize AFTER split: fit on train only
    train_mean = X_train.mean()
    train_std = X_train.std()

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    return X_train.values, X_test.values, y_train, y_test
