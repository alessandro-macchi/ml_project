import os
import numpy as np
from tests.test_src.test_preprocessing import load_and_combine_data, preprocess_features
from src.logistic_regression import LogisticRegressionScratch
from src.svm import SVMClassifierScratch
from src.kernels import KernelLogisticRegression, KernelPegasosSVM, create_named_kernels
from src.hyperparameter_tuning import grid_search
from sklearn.linear_model import LogisticRegression  # benchmark
from sklearn.svm import SVC  # benchmark
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, \
    confusion_matrix


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

    ksvm_param_grid = {
        "kernel_fn": named_kernels[:3],  # Limit to first 3 kernels for speed
        "lambda_": [0.01, 0.1],
        "max_iter": [500]
    }

    print("🔍 Grid search for Kernel SVM...")
    best_params_ksvm, best_score_ksvm = grid_search(X_train, y_train, KernelPegasosSVM, ksvm_param_grid)
    print(f"✅ Best KSVM params: {best_params_ksvm}, CV Accuracy: {best_score_ksvm:.4f}")

    # Train best model
    ksvm = KernelPegasosSVM(
        kernel_fn=best_params_ksvm["kernel_fn"],
        lambda_=best_params_ksvm["lambda_"],
        max_iter=best_params_ksvm["max_iter"]
    )
    ksvm.fit(X_train, y_train)
    pred_ksvm = ksvm.predict(X_test)

    results['ksvm_scratch'] = comprehensive_evaluation(y_test, pred_ksvm, "Kernel SVM (Scratch)")
    print(f"📊 Number of support vectors: {len(ksvm.support_vectors)}")

    # Sklearn kernel benchmarks
    rbf_svm_sk = SVC(kernel='rbf', C=1.0)
    rbf_svm_sk.fit(X_train, y_train)
    pred_rbf_svm_sk = rbf_svm_sk.predict(X_test)

    poly_svm_sk = SVC(kernel='poly', C=1.0, degree=3)
    poly_svm_sk.fit(X_train, y_train)
    pred_poly_svm_sk = poly_svm_sk.predict(X_test)

    results['rbf_svm_sklearn'] = comprehensive_evaluation(y_test, pred_rbf_svm_sk, "RBF SVM (sklearn)")
    results['poly_svm_sklearn'] = comprehensive_evaluation(y_test, pred_poly_svm_sk, "Poly SVM (sklearn)")

    return results


def print_comparison_table(results_baseline, results_smote):
    """Print a comparison table of results"""
    print(f"\n{'=' * 100}")
    print("📊 DETAILED COMPARISON: BASELINE vs SMOTE")
    print(f"{'=' * 100}")

    headers = ["Model", "Baseline Acc", "Baseline F1", "SMOTE Acc", "SMOTE F1", "F1 Improvement"]
    print(f"{headers[0]:<25} {headers[1]:<13} {headers[2]:<12} {headers[3]:<10} {headers[4]:<9} {headers[5]:<15}")
    print("-" * 100)

    model_names = {
        'lr_scratch': 'LR (Scratch)',
        'lr_sklearn': 'LR (sklearn)',
        'svm_scratch': 'SVM (Scratch)',
        'svm_sklearn': 'SVM (sklearn)',
        'klr_scratch': 'Kernel LR',
        'ksvm_scratch': 'Kernel SVM',
        'rbf_svm_sklearn': 'RBF SVM (sklearn)',
        'poly_svm_sklearn': 'Poly SVM (sklearn)'
    }

    for key in results_baseline.keys():
        if key in results_smote:
            baseline_acc = results_baseline[key]['accuracy']
            baseline_f1 = results_baseline[key]['f1']
            smote_acc = results_smote[key]['accuracy']
            smote_f1 = results_smote[key]['f1']
            f1_improvement = smote_f1 - baseline_f1

            name = model_names.get(key, key)
            improvement_str = f"+{f1_improvement:.4f}" if f1_improvement >= 0 else f"{f1_improvement:.4f}"

            print(
                f"{name:<25} {baseline_acc:<13.4f} {baseline_f1:<12.4f} {smote_acc:<10.4f} {smote_f1:<9.4f} {improvement_str:<15}")


def main():
    """Main function comparing baseline vs SMOTE approaches"""

    print("🍷 WINE QUALITY CLASSIFICATION: BASELINE vs SMOTE COMPARISON")
    print("=" * 80)

    # Load data
    red_path = os.path.join("../data", "winequality-red.csv")
    white_path = os.path.join("../data", "winequality-white.csv")
    data = load_and_combine_data(red_path, white_path)

    # Run experiments
    print("\n🚀 Starting experiments...")

    # Baseline experiment (no SMOTE)
    results_baseline = run_experiment(data, use_smote=False, experiment_name="BASELINE (No SMOTE)")

    # SMOTE experiment
    results_smote = run_experiment(data, use_smote=True, experiment_name="SMOTE OVERSAMPLING")

    # Print detailed comparison
    print_comparison_table(results_baseline, results_smote)

    # Summary recommendations
    print(f"\n{'=' * 80}")
    print("🎯 RECOMMENDATIONS")
    print(f"{'=' * 80}")

    # Find best F1 improvements
    f1_improvements = {}
    for key in results_baseline.keys():
        if key in results_smote:
            improvement = results_smote[key]['f1'] - results_baseline[key]['f1']
            f1_improvements[key] = improvement

    best_improvement = max(f1_improvements.items(), key=lambda x: x[1])
    worst_improvement = min(f1_improvements.items(), key=lambda x: x[1])

    print(f"✅ Best F1 improvement with SMOTE: {best_improvement[0]} (+{best_improvement[1]:.4f})")
    print(f"⚠️  Worst F1 change with SMOTE: {worst_improvement[0]} ({worst_improvement[1]:.4f})")

    # Calculate average improvements
    avg_f1_improvement = np.mean(list(f1_improvements.values()))
    print(f"📈 Average F1 improvement: {avg_f1_improvement:.4f}")

    if avg_f1_improvement > 0.01:
        print("🎉 SMOTE shows significant improvement overall!")
    elif avg_f1_improvement > 0:
        print("👍 SMOTE shows modest improvement overall.")
    else:
        print("🤔 SMOTE may not be beneficial for this dataset.")

    print(f"\n💡 Consider using SMOTE for models showing significant F1 improvement (>0.01)")
    print(f"💡 For production, use balanced accuracy and F1-score rather than accuracy")
    print(f"💡 Always validate on a separate hold-out test set not used in hyperparameter tuning")


if __name__ == "__main__":
    main()