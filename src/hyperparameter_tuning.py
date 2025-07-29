from src.cross_validation import cross_validate

def grid_search(X, y, model_class, param_grid, k_folds=5):
    """
    Generic grid search using cross-validation.

    Parameters:
        X, y: Dataset
        model_class: Model class to instantiate
        param_grid: Dictionary of parameter lists, e.g. {'lambda_': [...], 'max_iter': [...]}
        k_folds: Number of CV folds

    Returns:
        best_params, best_score
    """
    best_score = -1
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
        model_params = {k: v for k, v in params.items() if k != "max_iter" and k != "epochs"}
        epochs = params.get("max_iter") or params.get("epochs", 1000)

        try:
            score = cross_validate(X, y, model_class, model_params, k_folds=k_folds, epochs=epochs)
            print(f"✅ Tried {params} → CV Accuracy: {score:.4f}")

            if score > best_score:
                best_score = score
                best_params = params.copy()
        except Exception as e:
            print(f"⚠️ Skipped {params} due to error: {e}")

    if best_params is None:
        print("❌ All parameter combinations failed in grid search.")

    return best_params, best_score