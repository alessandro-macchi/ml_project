import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")


class OverfittingAnalyzer:
    """Streamlined overfitting analysis focused on learning curves"""

    def __init__(self, save_dir=None):
        """Initialize analyzer with directory management"""
        if save_dir is None:
            try:
                from utils.directory_management import get_directory_manager
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
        self.learning_curves = {}
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def analyze_all_models(self, models_dict, X_train, y_train, X_test, y_test, model_names=None):
        """Analyze learning curves for all models"""
        print("\n📊 LEARNING CURVE ANALYSIS")
        print("=" * 40)

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

        # Add models with proper naming
        for model_key, model in models_dict.items():
            model_name = self._get_model_display_name(model_key, model_names)
            self.models[model_key] = {'model': model, 'name': model_name}

        # Generate learning curves for each model
        for model_key in self.models.keys():
            print(f"\n📈 Generating learning curves for {self.models[model_key]['name']}...")
            self.learning_curves[model_key] = self._generate_learning_curves(model_key)

        print("\n✅ Learning curve analysis complete!")

    def _get_model_display_name(self, model_key, model_names):
        """Get display name for model with fallback logic"""
        if model_names and model_key in model_names:
            return model_names[model_key]

        name_mapping = {
            'lr_custom': 'Logistic Regression (Custom)',
            'svm_custom': 'Linear SVM (Custom)',
            'klr_custom': 'Kernel Logistic Regression (Custom)',
            'ksvm_custom': 'Kernel SVM (Custom)'
        }
        return name_mapping.get(model_key, model_key.replace('_', ' ').title())

    def _generate_learning_curves(self, model_key):
        """Generate learning curves to analyze overfitting"""
        try:
            model = self.models[model_key]['model']
            train_sizes = np.linspace(0.1, 1.0, 10)

            train_scores = []
            val_scores = []
            actual_train_sizes = []

            n_samples = len(self.X_train)

            for train_size in train_sizes:
                n_train = int(train_size * n_samples)
                if n_train < 10:
                    continue

                # Sample training data
                indices = np.random.choice(n_samples, n_train, replace=False)
                X_subset = self.X_train[indices]
                y_subset = self.y_train[indices]

                try:
                    # Create fresh model instance
                    subset_model = self._create_fresh_model(model_key, model)
                    if hasattr(subset_model, 'fit'):
                        subset_model.fit(X_subset, y_subset)

                    # Evaluate on training subset and validation set
                    train_pred = subset_model.predict(X_subset)
                    val_pred = subset_model.predict(self.X_test)

                    train_acc = accuracy_score(y_subset, train_pred)
                    val_acc = accuracy_score(self.y_test, val_pred)

                    train_scores.append(train_acc)
                    val_scores.append(val_acc)
                    actual_train_sizes.append(n_train)

                except Exception as e:
                    print(f"       ⚠️ Error at train_size {train_size}: {e}")
                    continue

            if len(train_scores) >= 3:
                return {
                    'available': True,
                    'train_sizes': np.array(actual_train_sizes),
                    'train_scores': np.array(train_scores),
                    'val_scores': np.array(val_scores)
                }
            else:
                return {'available': False, 'error': 'Insufficient learning curve data'}

        except Exception as e:
            print(f"     ⚠️ Learning curve generation failed: {e}")
            return {'available': False, 'error': str(e)}

    def _create_fresh_model(self, model_key, original_model):
        """Return a model with exactly the same parameters as the tuned/trained model"""
        try:
            import copy
            return copy.deepcopy(original_model)
        except Exception as e:
            print(f"⚠️ Could not copy model {model_key}: {e}")
            return original_model

    def _save_plot_safely(self, filename, dpi=300, bbox_inches='tight'):
        """Save plot with timestamp and proper error handling"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_filename = f"{filename}_{timestamp}.png"
            filepath = os.path.join(self.save_dir, full_filename)

            plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches,
                       facecolor='white', edgecolor='none')
            print(f"   💾 Saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"   ❌ Error saving {filename}: {e}")
            return None

    def plot_learning_curves(self, figsize=(16, 12), save_plots=False):
        """Create detailed learning curves for all models"""
        print("📈 Creating Learning Curves...")

        # Filter models with learning curve data
        models_with_curves = {
            k: v for k, v in self.learning_curves.items()
            if v['available']
        }

        if not models_with_curves:
            print("❌ No learning curve data available")
            return

        n_models = len(models_with_curves)
        cols = min(n_models, 2)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 1 and n_models > 1:
            axes = list(axes)
        else:
            axes = axes.flatten() if n_models > 1 else [axes]

        for i, (model_key, learning_curve) in enumerate(models_with_curves.items()):
            if i >= len(axes):
                break

            ax = axes[i]
            train_sizes = learning_curve['train_sizes']
            train_scores = learning_curve['train_scores']
            val_scores = learning_curve['val_scores']

            # Plot learning curves
            ax.plot(train_sizes, train_scores, 'o-', color='blue', linewidth=3,
                    markersize=6, label='Training Score', alpha=0.8)
            ax.plot(train_sizes, val_scores, 'o-', color='red', linewidth=3,
                    markersize=6, label='Validation Score', alpha=0.8)

            # Fill area between curves to show gap
            ax.fill_between(train_sizes, train_scores, val_scores,
                            alpha=0.2, color='red' if np.mean(train_scores) > np.mean(val_scores) + 0.05 else 'green')

            # Add trend lines
            if len(train_sizes) > 3:
                z_train = np.polyfit(train_sizes, train_scores, 1)
                p_train = np.poly1d(z_train)
                ax.plot(train_sizes, p_train(train_sizes), "--", alpha=0.8, color='blue', linewidth=1)

                z_val = np.polyfit(train_sizes, val_scores, 1)
                p_val = np.poly1d(z_val)
                ax.plot(train_sizes, p_val(train_sizes), "--", alpha=0.8, color='red', linewidth=1)

            # Formatting
            model_name = self.models[model_key]['name'].replace(' (Custom)', '')
            ax.set_xlabel('Training Set Size', fontweight='bold')
            ax.set_ylabel('Accuracy Score', fontweight='bold')
            ax.set_title(f'{model_name} Learning Curve', fontweight='bold', fontsize=14)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

            # Add gap annotation
            if len(train_scores) > 0 and len(val_scores) > 0:
                final_gap = train_scores[-1] - val_scores[-1]
                max_gap = np.max(train_scores - val_scores)

                gap_color = 'red' if final_gap > 0.1 else 'orange' if final_gap > 0.05 else 'green'
                ax.text(0.02, 0.98, f'Final Gap: {final_gap:.3f}\nMax Gap: {max_gap:.3f}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=gap_color, alpha=0.3),
                        fontsize=10, fontweight='bold')

        # Hide unused subplots
        for i in range(len(models_with_curves), len(axes)):
            if i < len(axes):
                axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle('Learning Curves Analysis', fontsize=16, fontweight='bold', y=1.02)

        if save_plots:
            self._save_plot_safely('learning_curves')
        plt.show()

    def create_learning_curve_analysis(self, figsize=(16, 12), save_plots=False):
        """Create learning curve analysis visualization"""
        print("\n🎨 CREATING LEARNING CURVE ANALYSIS")
        print("=" * 45)

        if not self.learning_curves:
            print("❌ No learning curve data available. Run analyze_all_models() first.")
            return

        if save_plots:
            print(f"💾 Plots will be saved to: {os.path.abspath(self.save_dir)}")

        print("\n📈 Generating learning curves...")
        self.plot_learning_curves(figsize, save_plots)

        print(f"\n🎉 Learning curve analysis complete!")


def integrate_overfitting_analysis(models_dict, X_train, y_train, X_test, y_test, model_names=None,
                                   save_plots=True, save_dir=None):
    """
    Integration function for existing project structure

    Args:
        models_dict (dict): Dictionary of trained models
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_names (dict): Optional display names for models
        save_plots (bool): Whether to save plots
        save_dir (str): Directory to save plots (if None, uses centralized manager)

    Returns:
        OverfittingAnalyzer: Analyzer with learning curve analysis
    """
    print("🔗 INTEGRATING LEARNING CURVE ANALYSIS")
    print("=" * 40)

    # Create analyzer with centralized directory management
    analyzer = OverfittingAnalyzer(save_dir=save_dir)

    # Run analysis
    analyzer.analyze_all_models(models_dict, X_train, y_train, X_test, y_test, model_names)

    # Create learning curve analysis
    analyzer.create_learning_curve_analysis(save_plots=save_plots)

    return analyzer