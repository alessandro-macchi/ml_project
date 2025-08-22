import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_roc_curves(self, X_test, y_test, figsize=(12, 8), save_plots=False):
    """Plot ROC curves for all models"""
    print("📈 Creating ROC Curves...")
    self._save_enabled = save_plots

    plt.figure(figsize=figsize)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for i, (model_key, model) in enumerate(self.models.items()):
        try:
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
            elif hasattr(model, 'decision_function'):
                y_scores = model.decision_function(X_test)
                y_proba = 1 / (1 + np.exp(-np.array(y_scores)))
            else:
                y_pred = model.predict(X_test)
                y_proba = y_pred.astype(float) + np.random.normal(0, 0.05, len(y_pred))
                y_proba = np.clip(y_proba, 0, 1)

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            color = colors[i % len(colors)]
            plt.plot(fpr, tpr, color=color, linewidth=2.5,
                     label=f'{self.model_names[model_key]} (AUC = {roc_auc:.3f})',
                     marker='o', markersize=3, alpha=0.8)

        except Exception as e:
            print(f"⚠️ Could not plot ROC for {model_key}: {e}")
            continue

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.50)', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_plots:
        self._save_figure("roc_curves")

    plt.tight_layout()
    plt.show()

def plot_precision_recall_curves(self, X_test, y_test, figsize=(12, 8), save_plots=False):
    """Plot Precision-Recall curves for all models"""
    print("📈 Creating Precision-Recall Curves...")
    self._save_enabled = save_plots

    plt.figure(figsize=figsize)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for i, (model_key, model) in enumerate(self.models.items()):
        try:
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
            elif hasattr(model, 'decision_function'):
                y_scores = model.decision_function(X_test)
                y_proba = 1 / (1 + np.exp(-np.array(y_scores)))
            else:
                y_pred = model.predict(X_test)
                y_proba = y_pred.astype(float) + np.random.normal(0, 0.05, len(y_pred))
                y_proba = np.clip(y_proba, 0, 1)

            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)

            color = colors[i % len(colors)]
            plt.plot(recall, precision, color=color, linewidth=2.5,
                     label=f'{self.model_names[model_key]} (AUC = {pr_auc:.3f})',
                     marker='o', markersize=3, alpha=0.8)

        except Exception as e:
            print(f"⚠️ Could not plot PR curve for {model_key}: {e}")
            continue

    baseline = np.sum(y_test) / len(y_test)
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=2,
                label=f'Random Classifier (Baseline = {baseline:.3f})', alpha=0.7)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_plots:
        self._save_figure("precision_recall_curves")

    plt.tight_layout()
    plt.show()