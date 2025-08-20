import numpy as np
from sklearn.metrics import accuracy_score
from core.kernels import NamedKernel
import time


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

def grid_search(X, y, model_class, param_grid, k_folds=5):
    """
    Generic grid search using cross-validation with improved logging.

    Parameters:
        X, y: Dataset
        model_class: Model class to instantiate
        param_grid: Dictionary of parameter lists, e.g. {'lambda_': [...], 'max_iter': [...]}
                   For kernel models, can include 'gamma_values', 'degree_values', 'coef0_values'
                   which will be converted to named kernel objects automatically.
        k_folds: Number of CV folds

    Returns:
        best_params, best_score
    """
    best_score = -1
    best_params = None

    # Let NamedKernel handle the kernel parameter conversion
    param_grid = NamedKernel.prepare_param_grid(param_grid)

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

    # Count total combinations for progress tracking
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)

    print(f"🔍 Starting grid search with {total_combinations} parameter combinations...")
    print(f"📊 Using {k_folds}-fold cross-validation")

    combination_count = 0

    for params in generate_combinations(param_values):
        combination_count += 1
        model_params = {k: v for k, v in params.items() if k != "max_iter" and k != "epochs"}
        epochs = params.get("max_iter") or params.get("epochs", 1000)

        # Create user-friendly parameter display
        param_display = format_params_display(params)

        print(f"\n🧪 [{combination_count}/{total_combinations}] Testing: {param_display}")

        start_time = time.time()

        try:
            score = cross_validate(X, y, model_class, model_params, k_folds=k_folds, epochs=epochs)
            elapsed_time = time.time() - start_time

            print(f"   ✅ CV Accuracy: {score:.4f} (took {elapsed_time:.1f}s)")

            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"   🏆 NEW BEST! Current best score: {best_score:.4f}")

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"   ❌ Failed after {elapsed_time:.1f}s: {str(e)[:50]}...")

    if best_params is None:
        print("\n❌ All parameter combinations failed in grid search.")
        return None, 0

    print(f"\n🎉 Grid search completed!")
    print(f"🏆 Best parameters: {format_params_display(best_params)}")
    print(f"📊 Best CV score: {best_score:.4f}")

    return best_params, best_score


def format_params_display(params):
    """Create a user-friendly display of parameters"""
    formatted_parts = []

    for key, value in params.items():
        if key in ["kernel", "kernel_fn"] and hasattr(value, 'name'):
            # NamedKernel objects have a .name attribute
            formatted_parts.append(f"{key}={value.name}")
        elif isinstance(value, float):
            formatted_parts.append(f"{key}={value:.4f}")
        else:
            formatted_parts.append(f"{key}={value}")

    return ", ".join(formatted_parts)