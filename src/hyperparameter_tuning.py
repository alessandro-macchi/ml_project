from itertools import product
from src.cross_validation import cross_validate

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

