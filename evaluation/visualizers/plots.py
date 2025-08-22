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
                from utils import get_directory_manager
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

    def plot_misclassifications(self, X_test, y_test, figsize=(16, 12), save_plots=False):
        """
        Analyze misclassified examples to understand model limitations

        Args:
            X_test: Test features
            y_test: Test labels
            figsize: Figure size for the plot
            save_plots (bool): Whether to save plots

        Returns:
            str: Path to saved plot or None
        """
        print("🔍 Creating Misclassification Analysis...")
        self._save_enabled = save_plots

        # Convert data to arrays
        if hasattr(X_test, 'values'):
            X_test_array = X_test.values
            feature_names = X_test.columns.tolist() if hasattr(X_test, 'columns') else None
        else:
            X_test_array = np.array(X_test)
            feature_names = None

        if hasattr(y_test, 'values'):
            y_test_array = y_test.values
        else:
            y_test_array = np.array(y_test)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_test_array.shape[1])]

        # Analyze each model
        n_models = len(self.models)
        if n_models == 0:
            print("❌ No models available for misclassification analysis")
            return None

        # Create subplot grid
        cols = 2
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if n_models == 1:
            axes = [axes]
        elif rows == 1 and n_models > 1:
            axes = list(axes)
        else:
            axes = axes.flatten() if n_models > 1 else [axes]

        model_keys = list(self.models.keys())

        for i, (model_key, model) in enumerate(self.models.items()):
            if i >= len(axes):  # Safety check
                break

            ax = axes[i]

            # Get predictions
            y_pred = model.predict(X_test)

            # Find misclassified examples
            misclassified_mask = (y_pred != y_test_array)

            if not np.any(misclassified_mask):
                ax.text(0.5, 0.5, 'No Misclassifications!',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, fontweight='bold')
                model_name = self.model_names.get(model_key, model_key).replace(' (Custom)', '')
                ax.set_title(f'{model_name}', fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Get misclassified examples
            X_misclassified = X_test_array[misclassified_mask]
            y_true_misc = y_test_array[misclassified_mask]
            y_pred_misc = y_pred[misclassified_mask]

            # Feature importance analysis for misclassified examples
            feature_importance = self._analyze_feature_patterns(
                X_misclassified, y_true_misc, y_pred_misc, feature_names
            )

            # Plot feature importance for misclassifications
            top_features = feature_importance[:8]  # Top 8 features
            feature_names_short = [name[:15] + '...' if len(name) > 15 else name
                                   for name, _ in top_features]
            importances = [imp for _, imp in top_features]

            if importances:  # Only plot if we have importance values
                bars = ax.barh(range(len(top_features)), importances, alpha=0.7)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(feature_names_short, fontsize=9)
                ax.set_xlabel('Misclassification Pattern Score', fontsize=10)

                # Add value labels
                max_importance = max(importances) if importances else 1
                for j, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + 0.01 * max_importance, bar.get_y() + bar.get_height() / 2,
                            f'{width:.2f}', ha='left', va='center', fontsize=8)

                ax.grid(True, alpha=0.3, axis='x')

            model_name = self.model_names.get(model_key, model_key).replace(' (Custom)', '')
            n_misc = len(X_misclassified)
            total = len(y_test_array)
            ax.set_title(f'{model_name}\n{n_misc}/{total} misclassified ({n_misc / total:.1%})',
                         fontweight='bold', fontsize=11)

        # Hide unused subplots
        for i in range(len(self.models), len(axes)):
            if i < len(axes):
                axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle('Misclassification Analysis - Feature Patterns',
                     fontsize=16, fontweight='bold', y=1.02)

        # Save plot
        saved_path = None
        if save_plots:
            saved_path = self._save_figure('misclassification_analysis')

        plt.show()

        # Print summary statistics
        self._print_misclassification_summary(X_test_array, y_test_array)

        return saved_path

    def _analyze_feature_patterns(self, X_misclassified, y_true, y_pred, feature_names):
        """Analyze feature patterns in misclassified examples"""
        feature_scores = []

        for i, feature_name in enumerate(feature_names):
            feature_values = X_misclassified[:, i]

            if len(feature_values) == 0:
                feature_scores.append((feature_name, 0.0))
                continue

            # Calculate various pattern indicators
            std_dev = np.std(feature_values)
            range_val = np.ptp(feature_values)  # peak-to-peak (max - min)
            mean_abs = np.mean(np.abs(feature_values))

            # Combine metrics for importance score
            # Higher variance and range often indicate problematic features
            importance_score = std_dev * 0.4 + range_val * 0.3 + mean_abs * 0.3

            feature_scores.append((feature_name, importance_score))

        # Sort by importance score (descending)
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        return feature_scores

    def _print_misclassification_summary(self, X_test_array, y_test_array):
        """Print summary statistics for misclassifications"""
        print("\n📊 MISCLASSIFICATION SUMMARY")
        print("-" * 50)

        for model_key, model in self.models.items():
            try:
                y_pred = model.predict(X_test_array)
                misclassified_mask = (y_pred != y_test_array)
                n_misc = np.sum(misclassified_mask)
                total = len(y_test_array)
                model_name = self.model_names.get(model_key, model_key)

                print(f"{model_name}: {n_misc}/{total} ({n_misc / total:.1%}) misclassified")
            except Exception as e:
                print(f"{model_key}: Error computing misclassifications - {e}")

    def plot_loss_curves(self, figsize=(12, 8), save_plots=False):
        """
        Plot training loss curves for all models that track losses

        Args:
            figsize: Figure size for the plot
            save_plots (bool): Whether to save plots

        Returns:
            str: Path to saved plot or None
        """
        print("📈 Creating Loss Curves Analysis...")
        self._save_enabled = save_plots

        # Extract models with loss tracking
        models_with_losses = {}
        for model_key, model in self.models.items():
            if hasattr(model, 'losses') and model.losses:
                models_with_losses[model_key] = model.losses

        if not models_with_losses:
            print("❌ No models with loss tracking found")
            return None

        # Create the plot
        plt.figure(figsize=figsize)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, (model_key, losses) in enumerate(models_with_losses.items()):
            # Get model display name
            model_name = self.model_names.get(model_key, model_key).replace(' (Custom)', '')
            color = colors[i % len(colors)]

            epochs = range(1, len(losses) + 1)
            plt.plot(epochs, losses, color=color, linewidth=2.5,
                     label=model_name, marker='o', markersize=2, alpha=0.8)

            # Add final loss annotation
            final_loss = losses[-1]
            plt.annotate(f'{final_loss:.4f}',
                         xy=(len(losses), final_loss),
                         xytext=(5, 0), textcoords="offset points",
                         ha='left', va='center', fontsize=9,
                         color=color, fontweight='bold')

        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title('Training Loss Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale often better for loss visualization

        plt.tight_layout()

        # Save plot
        saved_path = None
        if save_plots:
            saved_path = self._save_figure('loss_curves')

        plt.show()
        return saved_path

    # Updated create_essential_plots method to properly call the fixed functions
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
            print("\n1. Performance Metrics Comparison")
            self.plot_metrics_comparison(save_plots=save_plots)

            print("\n2. Confusion Matrices")
            self.plot_confusion_matrices(X_test, y_test, save_plots=save_plots)

            print("\n3. ROC Curves")
            self.plot_roc_curves(X_test, y_test, save_plots=save_plots)

            print("\n4. Precision-Recall Curves")
            self.plot_precision_recall_curves(X_test, y_test, save_plots=save_plots)

            print("\n5. Misclassification Analysis")
            self.plot_misclassifications(X_test, y_test, save_plots=save_plots)

            print("\n6. Loss Curves")
            self.plot_loss_curves(save_plots=save_plots)

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