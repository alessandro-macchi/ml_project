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
