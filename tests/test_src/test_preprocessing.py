import pandas as pd
import numpy as np

try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  SMOTE not available. Install with: pip install imbalanced-learn")


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


def check_class_balance(y, label="Dataset"):
    """Check and display class distribution"""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"📊 {label} Class Distribution:")
    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"   Class {cls}: {count:,} samples ({percentage:.1f}%)")

    if len(counts) == 2:
        imbalance_ratio = max(counts) / min(counts)
        print(f"   Imbalance Ratio: {imbalance_ratio:.2f}:1")
        return imbalance_ratio
    return None


def apply_smote_oversampling(X_train, y_train, random_state=42):
    """
    Apply SMOTE oversampling to training data only.

    CRITICAL: This function should ONLY be called on training data
    to avoid data leakage!
    """
    if not SMOTE_AVAILABLE:
        print("❌ SMOTE not available. Using original training data.")
        return X_train, y_train

    print("🔄 Applying SMOTE oversampling to training data...")

    # Check original distribution
    print("Before SMOTE:")
    check_class_balance(y_train, "Training")

    # Check if we have enough samples for SMOTE
    min_class_count = min(np.bincount(y_train))
    k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1

    if k_neighbors < 1:
        print("❌ Not enough samples in minority class for SMOTE. Using original data.")
        return X_train, y_train

    try:
        # Apply SMOTE
        smote = SMOTE(
            random_state=random_state,
            k_neighbors=k_neighbors,
            sampling_strategy='auto'  # Balance all classes to have same count as majority
        )
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print("After SMOTE:")
        check_class_balance(y_train_balanced, "Training (Balanced)")

        print(f"✅ SMOTE completed: {len(y_train):,} → {len(y_train_balanced):,} samples")

        return X_train_balanced, y_train_balanced

    except Exception as e:
        print(f"❌ SMOTE failed: {e}")
        print("   Using original training data.")
        return X_train, y_train


def preprocess_features(df: pd.DataFrame, apply_smote: bool = False, random_state: int = 6) -> tuple:
    """
    Preprocess wine dataset features and split into train/test sets.

    Args:
        df: Wine quality dataset
        apply_smote: Whether to apply SMOTE oversampling to training data
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test (with SMOTE applied to training data if requested)
    """
    # Create binary quality target
    df["quality_binary"] = (df["quality"] >= 6).astype(int)

    # Log-transform skewed features BEFORE splitting (this is safe)
    skewed_cols = ["residual sugar", "free sulfur dioxide", "total sulfur dioxide", "chlorides", "sulphates"]
    df = log_transform(df, skewed_cols)

    # Separate features and labels
    X = df.drop(columns=["quality", "quality_binary"])
    y = df["quality_binary"].values

    print(f"🔍 Original dataset shape: {X.shape}")
    check_class_balance(y, "Original Dataset")

    # STEP 1: Train-test split FIRST (critical to avoid data leakage)
    X_train, X_test, y_train, y_test = custom_train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    print(f"\n📊 After train-test split:")
    print(f"   Training set shape: {X_train.shape}")
    print(f"   Test set shape: {X_test.shape}")
    check_class_balance(y_train, "Training Set")
    check_class_balance(y_test, "Test Set")

    # STEP 2: Standardize features (fit on train, transform both)
    train_mean = X_train.mean()
    train_std = X_train.std()

    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std

    # Convert to numpy arrays
    X_train_scaled = X_train_scaled.values
    X_test_scaled = X_test_scaled.values

    # STEP 3: Apply SMOTE to training data ONLY (if requested)
    if apply_smote:
        print(f"\n{'=' * 50}")
        print("🧪 APPLYING SMOTE OVERSAMPLING")
        print(f"{'=' * 50}")
        X_train_scaled, y_train = apply_smote_oversampling(
            X_train_scaled, y_train, random_state=random_state
        )
        print(f"{'=' * 50}")
    else:
        print(f"\n📋 No SMOTE applied - using original training distribution")

    return X_train_scaled, X_test_scaled, y_train, y_test


def preprocess_features_original(df: pd.DataFrame) -> tuple:
    """Original preprocessing function without SMOTE (for backward compatibility)"""
    return preprocess_features(df, apply_smote=False)