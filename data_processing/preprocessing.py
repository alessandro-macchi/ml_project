import pandas as pd
import numpy as np
from .splitting import custom_train_test_split, generate_synthetic_samples


def log_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        df[col] = np.log1p(df[col])
    return df

def preprocess_features(df: pd.DataFrame, apply_smote=False) -> tuple:
    """Preprocess wine dataset features and split into train/test sets."""
    # Generate {-1, +1} labels instead of {0, 1}
    df["quality_binary"] = np.where(df["quality"] >= 6, 1, -1)

    # Separate features and labels before transformations
    X = df.drop(columns=["quality", "quality_binary"])
    y = df["quality_binary"].values

    # Train-test split first
    X_train, X_test, y_train, y_test = custom_train_test_split(
        X, y, test_size=0.2, random_state=6, stratify=y
    )

    # Log-transform skewed features after splitting
    skewed_cols = ["residual sugar", "free sulfur dioxide",
                   "total sulfur dioxide", "chlorides", "sulphates"]

    X_train = log_transform(X_train, skewed_cols)
    X_test = log_transform(X_test, skewed_cols)

    # Standardize after log transform
    train_mean = X_train.mean()
    train_std = X_train.std()

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    # Apply SMOTE oversampling
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