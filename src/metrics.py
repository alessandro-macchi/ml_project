from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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