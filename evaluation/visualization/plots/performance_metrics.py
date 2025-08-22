import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_metrics_comparison(self, figsize=(15, 7), save_plots=False):
    """Create metrics comparison plots for accuracy, precision/recall, and f1-score"""
    print("📊 Creating Metrics Comparison...")
    self._save_enabled = save_plots

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Prepare data
    model_keys = list(self.results.keys())
    model_display_names = [self.model_names[key].replace(' (Custom)', '') for key in model_keys]

    accuracies = [self.results[key]['accuracy'] for key in model_keys]
    precisions = [self.results[key]['precision'] for key in model_keys]
    recalls = [self.results[key]['recall'] for key in model_keys]
    f1_scores = [self.results[key]['f1'] for key in model_keys]

    # Plot 1: Accuracy Comparison
    bars1 = axes[0].bar(range(len(model_keys)), accuracies, alpha=0.8, color='skyblue')
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Comparison', fontweight='bold')
    axes[0].set_xticks(range(len(model_keys)))
    axes[0].set_xticklabels(model_display_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)

    # Plot 2: Precision vs Recall
    x = np.arange(len(model_keys))
    width = 0.35
    bars2 = axes[1].bar(x - width / 2, precisions, width, label='Precision', alpha=0.8, color='lightgreen')
    bars3 = axes[1].bar(x + width / 2, recalls, width, label='Recall', alpha=0.8, color='orange')

    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Precision vs Recall', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_display_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(f'{height:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8)

    # Plot 3: F1-Score
    bars4 = axes[2].bar(range(len(model_keys)), f1_scores, alpha=0.8, color='mediumpurple')
    axes[2].set_xlabel('Models')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_title('F1-Score Comparison', fontweight='bold')
    axes[2].set_xticks(range(len(model_keys)))
    axes[2].set_xticklabels(model_display_names, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    for bar in bars4:
        height = bar.get_height()
        axes[2].annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)

    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_plots:
        self._save_figure("metrics_comparison")
    plt.show()

def plot_confusion_matrices(self, X_test, y_test, figsize=(12, 8), save_plots=False):
    """Plot 2x2 confusion matrices"""
    print("🔍 Creating Confusion Matrices...")
    self._save_enabled = save_plots

    n_models = len(self.models)
    cols = 2
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    elif rows == 1 and n_models > 1:
        axes = list(axes)
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    for i, (model_key, model) in enumerate(self.models.items()):
        if i >= len(axes):
            break

        ax = axes[i]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Plot heatmap with counts only
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Low Quality (0)', 'High Quality (1)'],
                    yticklabels=['Low Quality (0)', 'High Quality (1)'],
                    cbar=False)

        model_name = self.model_names[model_key]
        ax.set_title(f'{model_name}', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    # Hide unused subplots
    for i in range(len(self.models), len(axes)):
        if i < len(axes):
            axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)

    if save_plots:
        self._save_figure("confusion_matrices")
    plt.show()