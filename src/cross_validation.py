import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def cross_validate(X, y, model_class, model_params, k_folds=5, epochs=1000):
    # Ensure consistent data types
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = np.array(X)

    if hasattr(y, 'values'):
        y_array = y.values
    else:
        y_array = np.array(y)

    n = len(X_array)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k_folds
    scores = []

    print(f"     🔄 Cross-validation: ", end="", flush=True)

    for i in range(k_folds):
        val_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, y_train = X_array[train_idx], y_array[train_idx]
        X_val, y_val = X_array[val_idx], y_array[val_idx]

        try:
            model = model_class(**model_params)

            # Try common `fit()` variants
            try:
                model.fit(X_train, y_train, max_iter=epochs)
            except TypeError:
                try:
                    model.fit(X_train, y_train, epochs=epochs)
                except TypeError:
                    model.fit(X_train, y_train)

            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            scores.append(acc)
            print(f"fold{i + 1}({acc:.3f}) ", end="", flush=True)

        except Exception as e:
            print(f"\n     ❌ Fold {i + 1} failed: {repr(e)}")
            return None  # Important for grid search to skip it

    print("✓")
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"     📊 Folds: {[f'{s:.3f}' for s in scores]} → μ={mean_score:.4f} ±{std_score:.3f}")

    return mean_score