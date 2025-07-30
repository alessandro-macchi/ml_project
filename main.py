import os
from src.preprocessing import load_and_combine_data, preprocess_features
from src.logistic_regression import LogisticRegressionScratch
from src.svm import SVMClassifierScratch
from src.kernels import KernelLogisticRegression, KernelPegasosSVM, create_named_kernels
from src.hyperparameter_tuning import grid_search
from sklearn.linear_model import LogisticRegression  # benchmark
from sklearn.svm import SVC  # benchmark
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score , confusion_matrix# evaluation


def comprehensive_evaluation(y_true, y_pred, model_name="Model"):
    """Comprehensive evaluation for imbalanced classification"""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"✅ {model_name} Results:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Balanced Accuracy: {bal_acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"   Confusion Matrix: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def run_experiment(data, use_smote=False, experiment_name="Baseline"):
    """Run a complete experiment with or without SMOTE"""

    print(f"\n{'=' * 70}")
    print(f"🧪 EXPERIMENT: {experiment_name}")
    print(f"{'=' * 70}")

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_features(data, apply_smote=use_smote)

    results = {}

    # 1. LOGISTIC REGRESSION
    print(f"\n{'=' * 50}")
    print("🔍 LOGISTIC REGRESSION")
    print(f"{'=' * 50}")

    lr_param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "regularization_strength": [0.0, 0.01, 0.1],
        "epochs": [500, 1000]
    }

    print("🔍 Grid search for Logistic Regression...")
    best_params_lr, best_score_lr = grid_search(X_train, y_train, LogisticRegressionScratch, lr_param_grid)
    print(f"✅ Best LR params: {best_params_lr}, CV Accuracy: {best_score_lr:.4f}")

    # Train best model
    model_lr = LogisticRegressionScratch(
        learning_rate=best_params_lr["learning_rate"],
        regularization_strength=best_params_lr["regularization_strength"],
        epochs=best_params_lr["epochs"]
    )
    model_lr.fit(X_train, y_train)
    pred_lr = model_lr.predict(X_test)

    # Sklearn benchmark
    model_lr_sk = LogisticRegression(penalty=None, solver='lbfgs', max_iter=best_params_lr["epochs"])
    model_lr_sk.fit(X_train, y_train)
    pred_lr_sk = model_lr_sk.predict(X_test)

    results['lr_scratch'] = comprehensive_evaluation(y_test, pred_lr, "Logistic Regression (Scratch)")
    results['lr_sklearn'] = comprehensive_evaluation(y_test, pred_lr_sk, "Logistic Regression (sklearn)")

    # 2. LINEAR SVM
    print(f"\n{'=' * 50}")
    print("🔍 LINEAR SVM")
    print(f"{'=' * 50}")

    svm_param_grid = {
        "lambda_": [0.001, 0.01, 0.1],
        "max_iter": [500, 1000]
    }

    print("🔍 Grid search for Linear SVM...")
    best_params_svm, best_score_svm = grid_search(X_train, y_train, SVMClassifierScratch, svm_param_grid)
    print(f"✅ Best SVM params: {best_params_svm}, CV Accuracy: {best_score_svm:.4f}")

    # Train best model
    svm_model = SVMClassifierScratch(lambda_=best_params_svm["lambda_"])
    svm_model.fit(X_train, y_train, max_iter=best_params_svm["max_iter"])
    pred_svm = svm_model.predict(X_test)

    # Sklearn benchmark
    l_svm_sk = SVC(kernel='linear', C=1.0)
    l_svm_sk.fit(X_train, y_train)
    pred_l_svm_sk = l_svm_sk.predict(X_test)

    results['svm_scratch'] = comprehensive_evaluation(y_test, pred_svm, "Linear SVM (Scratch)")
    results['svm_sklearn'] = comprehensive_evaluation(y_test, pred_l_svm_sk, "Linear SVM (sklearn)")

    # 3. KERNEL LOGISTIC REGRESSION
    print(f"\n{'=' * 50}")
    print("🔍 KERNEL LOGISTIC REGRESSION")
    print(f"{'=' * 50}")

    # Create kernel functions (reduced for speed)
    gamma_values = [0.01, 0.1]
    degree_values = [] #2, 3
    coef0_values = [] #0, 1
    named_kernels = create_named_kernels(gamma_values, degree_values, coef0_values)

    klr_param_grid = {
        "kernel": named_kernels,
        "lambda_": [0.01, 0.1],
        "epochs": [500]
    }

    print("🔍 Grid search for Kernel Logistic Regression...")
    best_params_klr, best_score_klr = grid_search(X_train, y_train, KernelLogisticRegression, klr_param_grid)
    print(f"✅ Best KLR params: {best_params_klr}, CV Accuracy: {best_score_klr:.4f}")

    # Train best model
    klr = KernelLogisticRegression(
        kernel=best_params_klr["kernel"],
        lambda_=best_params_klr["lambda_"],
        epochs=best_params_klr["epochs"],
        subsample_ratio=0.2,
        batch_size=64,
        early_stopping_patience=20
    )
    klr.fit(X_train, y_train)
    pred_klr = klr.predict(X_test)

    results['klr_scratch'] = comprehensive_evaluation(y_test, pred_klr, "Kernel Logistic Regression (Scratch)")

    # 4. KERNEL SVM
    print(f"\n{'=' * 50}")
    print("🔍 KERNEL SVM")
    print(f"{'=' * 50}")

    return results




def main():
    """Main function comparing baseline vs SMOTE approaches"""

    print("🍷 WINE QUALITY CLASSIFICATION: BASELINE vs SMOTE COMPARISON")
    print("=" * 80)

    # Load data
    red_path = os.path.join("data", "winequality-red.csv")
    white_path = os.path.join("data", "winequality-white.csv")
    data = load_and_combine_data(red_path, white_path)

    # Run experiments
    print("\n🚀 Starting experiments...")

    # Baseline experiment (no SMOTE)
    #results_baseline = run_experiment(data, use_smote=False, experiment_name="BASELINE (No SMOTE)")

    # SMOTE experiment
    results_smote = run_experiment(data, use_smote=True, experiment_name="SMOTE OVERSAMPLING")


if __name__ == "__main__":
    main()