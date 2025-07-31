import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def cross_validate(X, y, model_class, model_params, k_folds=5, epochs=1000):
    """
    Perform k-fold cross-validation with progress tracking.

    Args:
        X, y: Dataset (Pandas DataFrame/Series or NumPy arrays)
        model_class: Model class to instantiate
        model_params: Parameters to pass to model constructor
        k_folds: Number of folds
        epochs: Number of training epochs/iterations

    Returns:
        mean_accuracy: Average accuracy across folds
    """
    # Convert to DataFrame/Series if needed and reset index
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k_folds
    scores = []

    print(f"     🔄 Cross-validation: ", end="", flush=True)

    for i in range(k_folds):
        val_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = model_class(**model_params)

        # Fit depending on model interface
        if hasattr(model, 'fit'):
            try:
                model.fit(X_train, y_train, max_iter=epochs)
            except TypeError:
                model.fit(X_train, y_train)

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)

        print(f"fold{i + 1}({acc:.3f}) ", end="", flush=True)

    print("✓")
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"     📊 Folds: {[f'{s:.3f}' for s in scores]} → μ={mean_score:.4f} ±{std_score:.3f}")

    return mean_score
