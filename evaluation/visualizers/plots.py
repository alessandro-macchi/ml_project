"""
This module provides essential visualization functions for machine learning model evaluation:
1. Training loss curves
2. Performance metrics comparison
3. Confusion matrices
4. ROC curves comparison
5. Precision-recall curves comparison
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")


class ModelVisualizer:
    """Streamlined visualization class for essential ML model evaluation plots"""

    def __init__(self, save_dir=None):
        """Initialize the visualizer with directory management"""
        if save_dir is None:
            try:
                from src.save import get_directory_manager
                dir_manager = get_directory_manager()
                self.save_dir = dir_manager.plots_dir
                print(f"📁 Using centralized plots directory: {self.save_dir}")
            except ImportError:
                self.save_dir = "output/evaluation_plots"
                os.makedirs(self.save_dir, exist_ok=True)
                print(f"📁 Using fallback directory: {self.save_dir}")
        else:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"📁 Using provided directory: {self.save_dir}")

        self.models = {}
        self.results = {}
        self.model_names = {}

    def _save_figure(self, filename, dpi=300, bbox_inches='tight'):
        """Save figure with timestamp"""
        if not hasattr(self, '_save_enabled') or not self._save_enabled:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.png"
        filepath = os.path.join(self.save_dir, full_filename)

        try:
            plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches,
                       facecolor='white', edgecolor='none')
            print(f"💾 Saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error saving {filepath}: {e}")
            return None

    def add_model_results(self, model_key, trained_model, evaluation_results, model_name=None):
        """Add a trained model and its results for visualization"""
        self.models[model_key] = trained_model
        self.results[model_key] = evaluation_results

        if model_name is None:
            model_name = model_key.replace('_', ' ').title().replace('Custom', '(Custom)')
        self.model_names[model_key] = model_name

    def plot_metrics_comparison(self, figsize=(15, 5), save_plots=False):
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

    def create_essential_plots(self, X_test, y_test, save_plots=True, save_dir=None):
        """Generate all essential visualization plots"""
        print("\n🎨 GENERATING ESSENTIAL MODEL VISUALIZATIONS")
        print("=" * 60)

        if save_dir:
            self.save_dir = save_dir
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                print(f"📁 Updated directory: {self.save_dir}")

        if save_plots:
            print(f"💾 Plots will be saved to: {os.path.abspath(self.save_dir)}")

        try:
            print("\n2. Performance Metrics Comparison")
            self.plot_metrics_comparison(save_plots=save_plots)

            print("\n3. Confusion Matrices")
            self.plot_confusion_matrices(X_test, y_test, save_plots=save_plots)

            print("\n4. ROC Curves")
            self.plot_roc_curves(X_test, y_test, save_plots=save_plots)

            print("\n5. Precision-Recall Curves")
            self.plot_precision_recall_curves(X_test, y_test, save_plots=save_plots)

            print("\n✅ All essential visualizations generated successfully!")

            if save_plots:
                print(f"📁 All plots saved in: {os.path.abspath(self.save_dir)}")

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")


def create_model_visualizations(models_dict, results_dict, X_test, y_test, model_names=None,
                              save_plots=True, save_dir=None):
    """
    Convenience function to create all essential visualizations

    Args:
        models_dict (dict): Dictionary of trained models {model_key: model_object}
        results_dict (dict): Dictionary of evaluation results {model_key: results_dict}
        X_test: Test features
        y_test: Test labels
        model_names (dict): Optional custom model names {model_key: display_name}
        save_plots (bool): Whether to save plots (default: True)
        save_dir (str): Directory to save plots (if None, uses centralized manager)

    Returns:
        ModelVisualizer: The visualizer object
    """
    visualizer = ModelVisualizer(save_dir=save_dir)

    for model_key in models_dict.keys():
        if model_key in results_dict:
            model_name = None
            if model_names and model_key in model_names:
                model_name = model_names[model_key]

            visualizer.add_model_results(
                model_key,
                models_dict[model_key],
                results_dict[model_key],
                model_name
            )

    visualizer.create_essential_plots(X_test, y_test, save_plots=save_plots)
    return visualizer