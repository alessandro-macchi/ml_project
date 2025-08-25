import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

from .plots.diagnostic_plots import plot_roc_curves, plot_precision_recall_curves
from .plots.feature_analysis import plot_misclassifications, _analyze_feature_patterns, _print_misclassification_summary
from .plots.loss_curves import plot_loss_curves
from .plots.performance_metrics import plot_metrics_comparison, plot_confusion_matrices

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")


class ModelVisualizer:
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
        self.models[model_key] = trained_model
        self.results[model_key] = evaluation_results

        if model_name is None:
            model_name = model_key.replace('_', ' ').title().replace('Custom', '(Custom)')
        self.model_names[model_key] = model_name

    def plot_roc_curves(self, X_test, y_test, figsize=(12, 8), save_plots=False):
        """Plot ROC curves for all models"""
        return plot_roc_curves(self, X_test, y_test, figsize, save_plots)

    def plot_precision_recall_curves(self, X_test, y_test, figsize=(12, 8), save_plots=False):
        """Plot Precision-Recall curves for all models"""
        return plot_precision_recall_curves(self, X_test, y_test, figsize, save_plots)

    def plot_misclassifications(self, X_test, y_test, figsize=(16, 12), save_plots=False):
        """Analyze misclassified examples to understand model limitations"""
        return plot_misclassifications(self, X_test, y_test, figsize, save_plots)

    def plot_loss_curves(self, figsize=(12, 8), save_plots=False):
        """Plot training loss curves for all models that track losses"""
        return plot_loss_curves(self, figsize, save_plots)

    def plot_metrics_comparison(self, figsize=(15, 7), save_plots=False):
        """Create metrics comparison plots for accuracy, precision/recall, and f1-score"""
        return plot_metrics_comparison(self, figsize, save_plots)

    def plot_confusion_matrices(self, X_test, y_test, figsize=(12, 8), save_plots=False):
        """Plot 2x2 confusion matrices"""
        return plot_confusion_matrices(self, X_test, y_test, figsize, save_plots)

    def _analyze_feature_patterns(self, X_misclassified, y_true, y_pred, feature_names):
        """Analyze feature patterns in misclassified examples"""
        return _analyze_feature_patterns(self, X_misclassified, y_true, y_pred, feature_names)

    def _print_misclassification_summary(self, X_test_array, y_test_array):
        """Print summary statistics for misclassifications"""
        return _print_misclassification_summary(self, X_test_array, y_test_array)

    def create_essential_plots(self, X_test, y_test, save_plots=True, save_dir=None):
        print("\n🎨 Generating Model Visualizations")
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