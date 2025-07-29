import numpy as np
from sklearn.metrics import accuracy_score

def old_cross_validate(X, y, model_class, model_params, k_folds=5, epochs=1000):
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k_folds
    scores = []

    for i in range(k_folds):
        val_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)

    return np.mean(scores)


def cross_validate(X, y, model_class, model_params, k_folds=5, epochs=1000):
    """
    Perform k-fold cross-validation with progress tracking.

    Args:
        X, y: Dataset
        model_class: Model class to instantiate
        model_params: Parameters to pass to model constructor
        k_folds: Number of folds
        epochs: Number of training epochs/iterations

    Returns:
        mean_accuracy: Average accuracy across folds
    """
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k_folds
    scores = []

    print(f"     🔄 Cross-validation: ", end="", flush=True)

    for i in range(k_folds):
        # Create fold indices
        val_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Create and train model
        model = model_class(**model_params)

        # Handle different fit signatures
        if hasattr(model, 'fit'):
            if 'epochs' in model_params or hasattr(model, 'epochs'):
                model.fit(X_train, y_train)  # Epochs already set in constructor
            else:
                # For SVM-like models that take max_iter in fit()
                model.fit(X_train, y_train, max_iter=epochs)

        # Make predictions
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)

        # Progress indicator
        print(f"fold{i + 1}({acc:.3f}) ", end="", flush=True)

    print("✓")  # End the progress line

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print(f"     📊 Folds: {[f'{s:.3f}' for s in scores]} → μ={mean_score:.4f} ±{std_score:.3f}")

    return mean_score