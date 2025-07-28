from itertools import product
import numpy as np
from src.cross_validation import cross_validate
from sklearn.metrics import accuracy_score


def grid_search(X, y, model_class, learning_rates, epoch_list, reg_list, k_folds=5):
    best_score = -1
    best_params = None

    for lr, epochs, reg in product(learning_rates, epoch_list, reg_list):
        params = {
            "learning_rate": lr,
            "regularization_strength": reg
        }
        score = cross_validate(X, y, model_class, params, k_folds=k_folds, epochs=epochs)
        print(f"Tested: lr={lr}, epochs={epochs}, reg={reg} -> CV accuracy: {score:.4f}")
        if score > best_score:
            best_score = score
            best_params = {"learning_rate": lr, "epochs": epochs, "regularization_strength": reg}

    return best_params, best_score

def grid_search_svm(X, y, model_class, lambda_list, max_iter_list, k_folds=5):
    best_score = -np.inf
    best_params = None

    for lambda_ in lambda_list:
        for max_iter in max_iter_list:
            params = {'lambda_': lambda_}
            try:
                score = cross_validate(
                    X, y, model_class, params, k_folds=k_folds, epochs=max_iter
                )
                print(f"✅ Tried lambda={lambda_}, max_iter={max_iter} → CV Accuracy: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_params = {'lambda_': lambda_, 'max_iter': max_iter}
            except Exception as e:
                print(f"⚠️ Failed with {{'lambda_': {lambda_}}}: {e}")

    return best_params, best_score


def grid_search_kernel(X, y, model_class, param_grid, k_folds=5):
    """
    Cross-validation for kernel models.

    Parameters:
        X, y: dataset
        model_class: a class like KernelPegasosSVM
        param_grid: dictionary of hyperparameters (lambda_, max_iter, etc.)
        k_folds: number of cross-validation folds

    Returns:
        (best_params, best_score)
    """
    best_score = 0
    best_params = None

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    def generate_combinations(values, current_combo=None, index=0):
        if current_combo is None:
            current_combo = {}
        if index == len(values):
            yield current_combo.copy()
            return
        for value in values[index]:
            current_combo[param_names[index]] = value
            yield from generate_combinations(values, current_combo, index + 1)

    for params in generate_combinations(param_values):
        scores = []
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        fold_size = len(X) // k_folds

        for k in range(k_folds):
            val_idx = indices[k * fold_size: (k + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Create and train model
            model = model_class(**params)
            model.fit(X_train, y_train)

            # Predict and evaluate
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            scores.append(acc)

        avg_score = np.mean(scores)
        print(f"✅ Tested {params} → CV Accuracy: {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()

    return best_params, best_score
