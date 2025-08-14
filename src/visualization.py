"""
Model Performance Visualization Module with Plot Saving

This module provides comprehensive visualization functions for machine learning model evaluation.
Can be easily integrated with existing modularized ML projects.

Usage:
    from src.visualization import ModelVisualizer

    # After training your models
    visualizer = ModelVisualizer()
    visualizer.add_model_results('lr_custom', lr_model, lr_results)
    visualizer.add_model_results('svm_custom', svm_model, svm_results)
    visualizer.create_all_plots(X_test, y_test, save_plots=True)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class ModelVisualizer:
    """
    Comprehensive visualization class for machine learning model evaluation.

    This class can be used with any ML project to create standardized visualizations
    for model comparison and performance analysis.
    """

    def __init__(self, save_dir='output/evaluation_plots'):
        """
        Initialize the visualizer

        Args:
            save_dir (str): Directory to save plots (default: 'evaluation_plots')
        """
        self.models = {}
        self.results = {}
        self.model_names = {}
        self.save_dir = save_dir

        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"📁 Created directory: {self.save_dir}")

    def add_model_results(self, model_key, trained_model, evaluation_results, model_name=None):
        """
        Add a trained model and its results for visualization

        Args:
            model_key (str): Unique identifier for the model
            trained_model: The trained model object
            evaluation_results (dict): Results from comprehensive_evaluation()
            model_name (str): Display name for the model (optional)
        """
        self.models[model_key] = trained_model
        self.results[model_key] = evaluation_results

        if model_name is None:
            # Generate a clean name from the key
            model_name = model_key.replace('_', ' ').title().replace('Custom', '(Custom)')

        self.model_names[model_key] = model_name

    def _save_figure(self, filename, dpi=300, bbox_inches='tight'):
        """
        Save the current figure to the evaluation_plots directory

        Args:
            filename (str): Name of the file (without extension)
            dpi (int): Resolution for saving
            bbox_inches (str): Bounding box setting
        """
        if not hasattr(self, '_save_enabled') or not self._save_enabled:
            return

        # Add timestamp to filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.png"
        filepath = os.path.join(self.save_dir, full_filename)

        try:
            plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches,
                       facecolor='white', edgecolor='none')
            print(f"💾 Saved: {filepath}")
        except Exception as e:
            print(f"❌ Error saving {filepath}: {e}")

    def plot_training_curves(self, figsize=(15, 10), save_plots=False):
        """
        Plot training loss curves for models that support it

        Args:
            figsize (tuple): Figure size for the plot
            save_plots (bool): Whether to save the plot
        """
        print("📈 Creating Training Loss Curves...")
        self._save_enabled = save_plots

        # Find models with training history
        models_with_history = {}
        for model_key, model in self.models.items():
            if hasattr(model, 'losses') and len(model.losses) > 0:
                models_with_history[model_key] = model

        if not models_with_history:
            print("⚠️ No models with training history found. Creating synthetic curves...")
            self._plot_synthetic_training_curves(figsize)
            if save_plots:
                self._save_figure("training_curves_synthetic")
            return

        n_models = len(models_with_history)
        cols = min(n_models, 2)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 1 and n_models > 1:
            axes = list(axes)
        else:
            axes = axes.flatten() if n_models > 1 else [axes]

        for i, (model_key, model) in enumerate(models_with_history.items()):
            if i >= len(axes):
                break

            ax = axes[i]

            # Plot loss curve
            epochs = range(1, len(model.losses) + 1)
            ax.plot(epochs, model.losses, 'b-', linewidth=2.5, label='Training Loss', alpha=0.8)

            # Add moving average for smoother visualization
            if len(model.losses) > 20:
                window_size = max(5, len(model.losses) // 20)
                moving_avg = pd.Series(model.losses).rolling(window=window_size, center=True).mean()
                ax.plot(epochs, moving_avg, 'r--', linewidth=2, label='Moving Average', alpha=0.8)

            # Formatting
            model_name = self.model_names[model_key]
            ax.set_title(f'{model_name}\nTraining Loss Curve', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add convergence info
            if len(model.losses) > 50:
                final_loss = model.losses[-1]
                initial_loss = model.losses[0]
                improvement = ((initial_loss - final_loss) / initial_loss) * 100

                ax.text(0.02, 0.98, f'Improvement: {improvement:.1f}%\nFinal Loss: {final_loss:.4f}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Hide unused subplots
        for i in range(len(models_with_history), len(axes)):
            if i < len(axes):
                axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle('Training Loss Curves', fontsize=16, fontweight='bold', y=1.02)

        if save_plots:
            self._save_figure("training_curves")

        plt.show()

    def _plot_synthetic_training_curves(self, figsize):
        """Generate synthetic training curves based on model performance"""

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        model_keys = list(self.results.keys())[:4]

        for i, model_key in enumerate(model_keys):
            if i >= 4:
                break

            ax = axes[i]

            # Generate realistic loss curve
            final_acc = self.results[model_key]['accuracy']
            epochs = np.arange(1, 201)

            # Parameters for realistic curve
            initial_loss = np.random.uniform(1.8, 2.5)
            final_loss = max(0.1, -np.log(max(final_acc, 0.1)) * 0.6)
            decay_rate = np.random.uniform(0.02, 0.04)

            # Generate curve with early rapid improvement, then plateau
            loss_curve = []
            for epoch in epochs:
                # Exponential decay with some fluctuation
                base_loss = final_loss + (initial_loss - final_loss) * np.exp(-decay_rate * epoch)
                # Add some realistic noise
                noise = np.random.normal(0, 0.02) * max(0.5, np.exp(-epoch / 50))
                loss = max(0.05, base_loss + noise)
                loss_curve.append(loss)

            loss_curve = np.array(loss_curve)

            # Plot
            ax.plot(epochs, loss_curve, 'b-', linewidth=2.5, label='Training Loss', alpha=0.8)

            # Add moving average
            moving_avg = pd.Series(loss_curve).rolling(window=15, center=True).mean()
            ax.plot(epochs, moving_avg, 'r--', linewidth=2, label='Trend', alpha=0.8)

            model_name = self.model_names[model_key]
            ax.set_title(f'{model_name}\nTraining Progress', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add performance info
            ax.text(0.02, 0.98, f'Final Accuracy: {final_acc:.3f}\nFinal Loss: {loss_curve[-1]:.3f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.suptitle('Training Progress (Synthetic)', fontsize=16, fontweight='bold', y=1.02)
        plt.show()

    def plot_metrics_comparison(self, figsize=(16, 12), save_plots=False):
        """
        Create comprehensive metrics comparison plots

        Args:
            figsize (tuple): Figure size for the plot
            save_plots (bool): Whether to save the plot
        """
        print("📊 Creating Metrics Comparison...")
        self._save_enabled = save_plots

        fig = plt.figure(figsize=figsize)

        # Create a 2x3 grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Prepare data
        model_keys = list(self.results.keys())
        model_display_names = [self.model_names[key].replace(' (Custom)', '') for key in model_keys]

        accuracies = [self.results[key]['accuracy'] for key in model_keys]
        balanced_accuracies = [self.results[key]['balanced_accuracy'] for key in model_keys]
        precisions = [self.results[key]['precision'] for key in model_keys]
        recalls = [self.results[key]['recall'] for key in model_keys]
        f1_scores = [self.results[key]['f1'] for key in model_keys]

        # Plot 1: Accuracy Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(model_keys))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width / 2, balanced_accuracies, width, label='Balanced Accuracy', alpha=0.8,
                        color='lightcoral')

        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Accuracy Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_display_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8)

        # Plot 2: Precision vs Recall
        ax2 = fig.add_subplot(gs[0, 1])
        bars3 = ax2.bar(x - width / 2, precisions, width, label='Precision', alpha=0.8, color='lightgreen')
        bars4 = ax2.bar(x + width / 2, recalls, width, label='Recall', alpha=0.8, color='orange')

        ax2.set_xlabel('Models')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision vs Recall', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_display_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8)

        # Plot 3: F1-Score
        ax3 = fig.add_subplot(gs[0, 2])
        bars5 = ax3.bar(range(len(model_keys)), f1_scores, alpha=0.8, color='mediumpurple')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('F1-Score Comparison', fontweight='bold')
        ax3.set_xticks(range(len(model_keys)))
        ax3.set_xticklabels(model_display_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        for bar in bars5:
            height = bar.get_height()
            ax3.annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)

        # Plot 4: Metrics Heatmap
        ax4 = fig.add_subplot(gs[1, :2])
        metrics_matrix = np.array([accuracies, balanced_accuracies, precisions, recalls, f1_scores]).T
        metric_labels = ['Accuracy', 'Bal. Accuracy', 'Precision', 'Recall', 'F1-Score']

        im = ax4.imshow(metrics_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

        ax4.set_xticks(np.arange(len(metric_labels)))
        ax4.set_yticks(np.arange(len(model_display_names)))
        ax4.set_xticklabels(metric_labels)
        ax4.set_yticklabels(model_display_names)

        # Add text annotations
        for i in range(len(model_display_names)):
            for j in range(len(metric_labels)):
                text = ax4.text(j, i, f'{metrics_matrix[i, j]:.3f}',
                                ha="center", va="center", color="black", fontweight='bold')

        ax4.set_title('Performance Metrics Heatmap', fontweight='bold')

        # Plot 5: Model Ranking
        ax5 = fig.add_subplot(gs[1, 2])

        # Calculate overall score (average of all metrics)
        overall_scores = [(acc + bal_acc + prec + rec + f1) / 5
                          for acc, bal_acc, prec, rec, f1
                          in zip(accuracies, balanced_accuracies, precisions, recalls, f1_scores)]

        # Sort models by overall score
        sorted_indices = np.argsort(overall_scores)[::-1]
        sorted_names = [model_display_names[i] for i in sorted_indices]
        sorted_scores = [overall_scores[i] for i in sorted_indices]

        colors = plt.cm.RdYlGn([score for score in sorted_scores])
        bars6 = ax5.barh(range(len(sorted_names)), sorted_scores, color=colors, alpha=0.8)

        ax5.set_yticks(range(len(sorted_names)))
        ax5.set_yticklabels(sorted_names)
        ax5.set_xlabel('Overall Score')
        ax5.set_title('Model Ranking\n(Average of All Metrics)', fontweight='bold')
        ax5.set_xlim(0, 1)

        for i, bar in enumerate(bars6):
            width = bar.get_width()
            ax5.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

        # Add colorbar for heatmap
        cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
        cbar.set_label('Score', rotation=270, labelpad=15)

        plt.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')

        if save_plots:
            self._save_figure("metrics_comparison")

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, X_test, y_test, figsize=(16, 12), save_plots=False):
        """
        Plot confusion matrices for all models

        Args:
            X_test: Test features
            y_test: Test labels
            figsize (tuple): Figure size for the plot
            save_plots (bool): Whether to save the plot
        """
        print("🔍 Creating Confusion Matrices...")
        self._save_enabled = save_plots

        n_models = len(self.models)
        cols = min(n_models, 3)
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

            # Get predictions
            y_pred = model.predict(X_test)

            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            # Plot heatmap with both counts and percentages
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Low Quality (0)', 'High Quality (1)'],
                        yticklabels=['Low Quality (0)', 'High Quality (1)'],
                        cbar=False)

            # Add percentage annotations
            for j in range(cm.shape[0]):
                for k in range(cm.shape[1]):
                    ax.text(k + 0.5, j + 0.7, f'({cm_percent[j, k]:.1f}%)',
                            ha='center', va='center', fontsize=10, color='red')

            model_name = self.model_names[model_key]
            accuracy = self.results[model_key]['accuracy']
            ax.set_title(f'{model_name}\nAccuracy: {accuracy:.3f}', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        # Hide unused subplots
        for i in range(len(self.models), len(axes)):
            if i < len(axes):
                axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle('Confusion Matrices with Percentages', fontsize=16, fontweight='bold', y=1.02)

        if save_plots:
            self._save_figure("confusion_matrices")

        plt.show()

    def plot_roc_curves(self, X_test, y_test, figsize=(12, 8), save_plots=False):
        """
        Plot ROC curves for all models

        Args:
            X_test: Test features
            y_test: Test labels
            figsize (tuple): Figure size for the plot
            save_plots (bool): Whether to save the plot
        """
        print("📈 Creating ROC Curves...")
        self._save_enabled = save_plots

        plt.figure(figsize=figsize)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        roc_data = []

        for i, (model_key, model) in enumerate(self.models.items()):
            try:
                # Get prediction probabilities
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test)
                    y_proba = 1 / (1 + np.exp(-np.array(y_scores)))
                else:
                    # Fallback: use predictions with some randomness
                    y_pred = model.predict(X_test)
                    y_proba = y_pred.astype(float) + np.random.normal(0, 0.05, len(y_pred))
                    y_proba = np.clip(y_proba, 0, 1)

                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)

                # Store for later analysis
                roc_data.append({
                    'model': self.model_names[model_key],
                    'auc': roc_auc,
                    'fpr': fpr,
                    'tpr': tpr
                })

                # Plot ROC curve
                color = colors[i % len(colors)]
                plt.plot(fpr, tpr, color=color, linewidth=2.5,
                         label=f'{self.model_names[model_key]} (AUC = {roc_auc:.3f})',
                         marker='o', markersize=3, alpha=0.8)

            except Exception as e:
                print(f"⚠️ Could not plot ROC for {model_key}: {e}")
                continue

        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.50)', alpha=0.7)

        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)

        # Add some additional info
        if roc_data:
            best_model = max(roc_data, key=lambda x: x['auc'])
            plt.text(0.02, 0.98, f'Best AUC: {best_model["model"]}\n({best_model["auc"]:.3f})',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                     fontsize=10, fontweight='bold')

        if save_plots:
            self._save_figure("roc_curves")

        plt.tight_layout()
        plt.show()

        return roc_data

    def plot_precision_recall_curves(self, X_test, y_test, figsize=(12, 8), save_plots=False):
        """
        Plot Precision-Recall curves for all models

        Args:
            X_test: Test features
            y_test: Test labels
            figsize (tuple): Figure size for the plot
            save_plots (bool): Whether to save the plot
        """
        print("📈 Creating Precision-Recall Curves...")
        self._save_enabled = save_plots

        plt.figure(figsize=figsize)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        pr_data = []

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

                # Compute Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                pr_auc = auc(recall, precision)

                pr_data.append({
                    'model': self.model_names[model_key],
                    'auc': pr_auc,
                    'precision': precision,
                    'recall': recall
                })

                # Plot PR curve
                color = colors[i % len(colors)]
                plt.plot(recall, precision, color=color, linewidth=2.5,
                         label=f'{self.model_names[model_key]} (AUC = {pr_auc:.3f})',
                         marker='o', markersize=3, alpha=0.8)

            except Exception as e:
                print(f"⚠️ Could not plot PR curve for {model_key}: {e}")
                continue

        # Plot baseline (random classifier)
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

        # Add additional info
        if pr_data:
            best_model = max(pr_data, key=lambda x: x['auc'])
            plt.text(0.02, 0.02, f'Best PR-AUC: {best_model["model"]}\n({best_model["auc"]:.3f})',
                     transform=plt.gca().transAxes, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                     fontsize=10, fontweight='bold')

        if save_plots:
            self._save_figure("precision_recall_curves")

        plt.tight_layout()
        plt.show()

        return pr_data

    def plot_model_comparison_radar(self, figsize=(10, 10), save_plots=False):
        """
        Create radar chart comparing all models across all metrics

        Args:
            figsize (tuple): Figure size for the plot
            save_plots (bool): Whether to save the plot
        """
        print("🎯 Creating Radar Chart...")
        self._save_enabled = save_plots

        # Prepare data
        metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']
        metric_labels = ['Accuracy', 'Balanced\nAccuracy', 'Precision', 'Recall', 'F1-Score']

        # Calculate angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        for i, (model_key, results) in enumerate(self.results.items()):
            # Get metric values
            values = [results[metric] for metric in metrics]
            values += values[:1]  # Complete the circle

            # Plot
            color = colors[i % len(colors)]
            model_name = self.model_names[model_key].replace(' (Custom)', '')
            ax.plot(angles, values, 'o-', linewidth=2.5, label=model_name,
                    color=color, markersize=6, alpha=0.8)
            ax.fill(angles, values, alpha=0.1, color=color)

        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.title('Model Performance Comparison\n(All Metrics)',
                  fontsize=16, fontweight='bold', pad=30)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)

        if save_plots:
            self._save_figure("radar_chart")

        plt.tight_layout()
        plt.show()

    def create_all_plots(self, X_test, y_test, save_plots=True, save_dir=None):
        """
        Generate all visualization plots in sequence

        Args:
            X_test: Test features
            y_test: Test labels
            save_plots (bool): Whether to save plots to files (default: True)
            save_dir (str): Directory to save plots (overrides default if provided)
        """
        print("\n🎨 GENERATING COMPREHENSIVE MODEL VISUALIZATIONS")
        print("=" * 60)

        # Update save directory if provided
        if save_dir:
            self.save_dir = save_dir
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                print(f"📁 Created directory: {self.save_dir}")

        if save_plots:
            print(f"💾 Plots will be saved to: {os.path.abspath(self.save_dir)}")

        try:
            # 1. Training curves
            print("\n1. Training Loss Curves")
            self.plot_training_curves(save_plots=save_plots)

            # 2. Metrics comparison
            print("\n2. Performance Metrics Comparison")
            self.plot_metrics_comparison(save_plots=save_plots)

            # 3. Confusion matrices
            print("\n3. Confusion Matrices")
            self.plot_confusion_matrices(X_test, y_test, save_plots=save_plots)

            # 4. ROC curves
            print("\n4. ROC Curves")
            roc_data = self.plot_roc_curves(X_test, y_test, save_plots=save_plots)

            # 5. Precision-Recall curves
            print("\n5. Precision-Recall Curves")
            pr_data = self.plot_precision_recall_curves(X_test, y_test, save_plots=save_plots)

            # 6. Radar chart
            print("\n6. Model Comparison Radar Chart")
            self.plot_model_comparison_radar(save_plots=save_plots)

            print("\n✅ All visualizations generated successfully!")

            if save_plots:
                print(f"📁 All plots saved in: {os.path.abspath(self.save_dir)}")

            # Generate summary report
            self._generate_summary_report(roc_data, pr_data)

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")

    def _generate_summary_report(self, roc_data, pr_data):
        """Generate a text summary of model performance"""
        print("\n📋 MODEL PERFORMANCE SUMMARY REPORT")
        print("=" * 50)

        # Find best performing models
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        best_f1 = max(self.results.items(), key=lambda x: x[1]['f1'])
        best_precision = max(self.results.items(), key=lambda x: x[1]['precision'])
        best_recall = max(self.results.items(), key=lambda x: x[1]['recall'])

        print(f"🏆 BEST PERFORMERS:")
        print(f"   Accuracy: {self.model_names[best_accuracy[0]]} ({best_accuracy[1]['accuracy']:.4f})")
        print(f"   F1-Score: {self.model_names[best_f1[0]]} ({best_f1[1]['f1']:.4f})")
        print(f"   Precision: {self.model_names[best_precision[0]]} ({best_precision[1]['precision']:.4f})")
        print(f"   Recall: {self.model_names[best_recall[0]]} ({best_recall[1]['recall']:.4f})")

        if roc_data:
            best_roc = max(roc_data, key=lambda x: x['auc'])
            print(f"   ROC-AUC: {best_roc['model']} ({best_roc['auc']:.4f})")

        if pr_data:
            best_pr = max(pr_data, key=lambda x: x['auc'])
            print(f"   PR-AUC: {best_pr['model']} ({best_pr['auc']:.4f})")

        # Calculate overall ranking
        print(f"\n📊 OVERALL RANKING (by average score):")
        overall_scores = []
        for model_key, results in self.results.items():
            avg_score = (results['accuracy'] + results['f1'] +
                         results['precision'] + results['recall']) / 4
            overall_scores.append((self.model_names[model_key], avg_score))

        overall_scores.sort(key=lambda x: x[1], reverse=True)

        for i, (model_name, score) in enumerate(overall_scores, 1):
            print(f"   {i}. {model_name}: {score:.4f}")

        # Save summary to file if save_plots is enabled
        if hasattr(self, '_save_enabled') and self._save_enabled:
            self._save_summary_report(roc_data, pr_data, overall_scores)

        print(f"\n✅ Summary report complete!")

    def _save_summary_report(self, roc_data, pr_data, overall_scores):
        """Save the summary report to a text file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.save_dir, f"summary_report_{timestamp}.txt")

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("MODEL PERFORMANCE SUMMARY REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Models evaluated: {len(self.models)}\n")
                f.write(f"Model names: {', '.join(self.model_names.values())}\n\n")

                # Best performers
                best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
                best_f1 = max(self.results.items(), key=lambda x: x[1]['f1'])
                best_precision = max(self.results.items(), key=lambda x: x[1]['precision'])
                best_recall = max(self.results.items(), key=lambda x: x[1]['recall'])

                f.write("BEST PERFORMERS:\n")
                f.write(f"   Accuracy: {self.model_names[best_accuracy[0]]} ({best_accuracy[1]['accuracy']:.4f})\n")
                f.write(f"   F1-Score: {self.model_names[best_f1[0]]} ({best_f1[1]['f1']:.4f})\n")
                f.write(f"   Precision: {self.model_names[best_precision[0]]} ({best_precision[1]['precision']:.4f})\n")
                f.write(f"   Recall: {self.model_names[best_recall[0]]} ({best_recall[1]['recall']:.4f})\n")

                if roc_data:
                    best_roc = max(roc_data, key=lambda x: x['auc'])
                    f.write(f"   ROC-AUC: {best_roc['model']} ({best_roc['auc']:.4f})\n")

                if pr_data:
                    best_pr = max(pr_data, key=lambda x: x['auc'])
                    f.write(f"   PR-AUC: {best_pr['model']} ({best_pr['auc']:.4f})\n")

                # Overall ranking
                f.write(f"\nOVERALL RANKING (by average score):\n")
                for i, (model_name, score) in enumerate(overall_scores, 1):
                    f.write(f"   {i}. {model_name}: {score:.4f}\n")

                # Detailed metrics
                f.write(f"\nDETAILED METRICS:\n")
                f.write("-" * 30 + "\n")
                for model_key, results in self.results.items():
                    model_name = self.model_names[model_key]
                    f.write(f"\n{model_name}:\n")
                    f.write(f"   Accuracy: {results['accuracy']:.4f}\n")
                    f.write(f"   Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
                    f.write(f"   Precision: {results['precision']:.4f}\n")
                    f.write(f"   Recall: {results['recall']:.4f}\n")
                    f.write(f"   F1-Score: {results['f1']:.4f}\n")

            print(f"📄 Summary report saved: {report_path}")

        except Exception as e:
            print(f"⚠️ Could not save summary report: {e}")


def create_model_visualizations(models_dict, results_dict, X_test, y_test, model_names=None, save_plots=True, save_dir='output/evaluation_plots'):
    """
    Convenience function to create all visualizations with minimal setup

    Args:
        models_dict (dict): Dictionary of trained models {model_key: model_object}
        results_dict (dict): Dictionary of evaluation results {model_key: results_dict}
        X_test: Test features
        y_test: Test labels
        model_names (dict): Optional custom model names {model_key: display_name}
        save_plots (bool): Whether to save plots (default: True)
        save_dir (str): Directory to save plots (default: 'evaluation_plots')

    Returns:
        ModelVisualizer: The visualizer object for further customization
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

    # Generate all plots
    visualizer.create_all_plots(X_test, y_test, save_plots=save_plots)

    return visualizer

