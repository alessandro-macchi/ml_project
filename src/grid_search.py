from src.cross_validation import cross_validate
import time


def grid_search(X, y, model_class, param_grid, k_folds=5):
    """
    Generic grid search using cross-validation with improved logging.

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
        if key == "kernel":
            # For kernel objects, show their name
            if hasattr(value, 'name'):
                formatted_parts.append(f"{key}={value.name}")
            elif hasattr(value, '__name__'):
                formatted_parts.append(f"{key}={value.__name__}")
            else:
                formatted_parts.append(f"{key}={str(value)[:20]}...")
        elif key == "kernel_fn":
            # For kernel function objects
            if hasattr(value, 'name'):
                formatted_parts.append(f"kernel={value.name}")
            else:
                formatted_parts.append(f"kernel={str(value)[:20]}...")
        elif isinstance(value, float):
            formatted_parts.append(f"{key}={value:.4f}")
        else:
            formatted_parts.append(f"{key}={value}")

    return ", ".join(formatted_parts)