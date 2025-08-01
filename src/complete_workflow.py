import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CompleteAnalysisWorkflow:
    """
    Complete workflow for wine quality classification analysis with SMOTE including:
    1. Data loading and preprocessing with SMOTE
    2. Custom model training with history tracking
    3. Comprehensive evaluation
    4. Misclassification analysis
    5. Results saving and visualization
    """

    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None

        self.models = {}
        self.results = {}
        self.training_histories = {}
        self.misclassification_analysis = {}

        print("🍷 Wine Quality Classification - Complete Analysis Workflow (SMOTE)")
        print("=" * 70)

    def load_and_preprocess_data(self, red_path="data/winequality-red.csv",
                                 white_path="data/winequality-white.csv"):
        """Load and preprocess the wine quality dataset with SMOTE"""

        print("\n📊 LOADING AND PREPROCESSING DATA WITH SMOTE")
        print("-" * 50)

        try:
            # Load data
            red_wine = pd.read_csv(red_path, sep=';')
            white_wine = pd.read_csv(white_path, sep=';')

            print(f"✅ Loaded {len(red_wine)} red wine samples")
            print(f"✅ Loaded {len(white_wine)} white wine samples")

            # Combine datasets
            red_wine['wine_type'] = 0  # red
            white_wine['wine_type'] = 1  # white
            self.data = pd.concat([red_wine, white_wine], ignore_index=True)

            # Create binary target
            self.data['quality_binary'] = (self.data['quality'] >= 6).astype(int)

            # Analyze class distribution
            class_dist = self.data['quality_binary'].value_counts()
            print(f"📈 Original class distribution:")
            print(f"   Low quality (0): {class_dist[0]} ({class_dist[0] / len(self.data) * 100:.1f}%)")
            print(f"   High quality (1): {class_dist[1]} ({class_dist[1] / len(self.data) * 100:.1f}%)")

            # Log transform skewed features
            skewed_features = ['residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'chlorides',
                               'sulphates']
            for feature in skewed_features:
                if feature in self.data.columns:
                    self.data[feature] = np.log1p(self.data[feature])

            # Prepare features and target
            X = self.data.drop(columns=['quality', 'quality_binary'])
            y = self.data['quality_binary'].values

            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Validation split from training data
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
            )

            # Standardization
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_val = scaler.transform(self.X_val)
            self.X_test = scaler.transform(self.X_test)

            # Always apply SMOTE
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print("✅ Applied SMOTE oversampling")

            # Show new class distribution
            unique, counts = np.unique(self.y_train, return_counts=True)
            print(f"📈 SMOTE class distribution:")
            for cls, count in zip(unique, counts):
                print(f"   Class {cls}: {count} ({count / len(self.y_train) * 100:.1f}%)")

            print(f"✅ Final training set: {len(self.X_train)} samples")
            print(f"✅ Validation set: {len(self.X_val)} samples")
            print(f"✅ Test set: {len(self.X_test)} samples")

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def train_custom_models(self):
        """Train custom models only"""

        print("\n🏋️ TRAINING CUSTOM MODELS")
        print("-" * 50)

        # Import custom models
        from models.logistic_regression import LogisticRegressionScratch
        from models.svm import SVMClassifierScratch
        from models.kernel_logistic_regression import KernelLogisticRegression
        from models.kernel_svm import KernelPegasosSVM
        from src.utils import create_named_kernels

        # Define custom models to train
        models_config = {
            'lr_custom': {
                'model_class': LogisticRegressionScratch,
                'params': {'learning_rate': 0.1, 'regularization_strength': 0.01, 'epochs': 1000},
                'name': 'Logistic Regression (Custom)'
            },
            'svm_custom': {
                'model_class': SVMClassifierScratch,
                'params': {'lambda_': 0.01},
                'name': 'Linear SVM (Custom)',
                'fit_params': {'max_iter': 1500}
            },
            'klr_custom': {
                'model_class': KernelLogisticRegression,
                'params': {
                    'kernel': create_named_kernels(gamma_values=[0.1], degree_values=[], coef0_values=[])[0],
                    'lambda_': 0.01,
                    'epochs': 600
                },
                'name': 'Kernel Logistic Regression (Custom)'
            },
            'ksvm_custom': {
                'model_class': KernelPegasosSVM,
                'params': {
                    'kernel': create_named_kernels(gamma_values=[0.1], degree_values=[3], coef0_values=[1])[1],
                    'lambda_': 0.01,
                    'max_iter': 1500
                },
                'name': 'Kernel SVM (Custom)'
            }
        }

        # Train each model
        for model_key, config in models_config.items():
            print(f"\n📈 Training {config['name']}...")

            model_class = config['model_class']
            params = config['params']
            fit_params = config.get('fit_params', {})

            try:
                model = model_class(**params)

                # Train model
                if fit_params:
                    model.fit(self.X_train, self.y_train, **fit_params)
                else:
                    model.fit(self.X_train, self.y_train)

                # Make predictions
                train_pred = model.predict(self.X_train)
                val_pred = model.predict(self.X_val)
                test_pred = model.predict(self.X_test)

                # Get prediction probabilities if available
                try:
                    train_proba = model.predict_proba(self.X_train)
                    val_proba = model.predict_proba(self.X_val)
                    test_proba = model.predict_proba(self.X_test)
                except:
                    train_proba = val_proba = test_proba = None

                # Calculate metrics
                train_metrics = self._calculate_metrics(self.y_train, train_pred)
                val_metrics = self._calculate_metrics(self.y_val, val_pred)
                test_metrics = self._calculate_metrics(self.y_test, test_pred)

                # Store model and results
                self.models[model_key] = {
                    'model': model,
                    'name': config['name'],
                    'train_pred': train_pred,
                    'val_pred': val_pred,
                    'test_pred': test_pred,
                    'train_proba': train_proba,
                    'val_proba': val_proba,
                    'test_proba': test_proba
                }

                self.results[model_key] = {
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics
                }

                print(
                    f"   ✅ Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")

            except Exception as e:
                print(f"   ❌ Failed to train {config['name']}: {e}")

        print("\n✅ All custom models trained successfully!")
        return True

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def analyze_misclassifications(self):
        """Perform detailed misclassification analysis"""

        print("\n🔍 MISCLASSIFICATION ANALYSIS")
        print("-" * 50)

        for model_key, model_data in self.models.items():
            print(f"\n📊 Analyzing {model_data['name']}...")

            # Get test predictions and true labels
            y_true = self.y_test
            y_pred = model_data['test_pred']

            # Find misclassified examples
            misclassified_mask = y_true != y_pred
            misclassified_indices = np.where(misclassified_mask)[0]

            # Analyze misclassification patterns
            false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
            false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]

            total_misclassified = len(misclassified_indices)
            total_samples = len(y_true)

            analysis = {
                'total_misclassified': total_misclassified,
                'misclassification_rate': total_misclassified / total_samples,
                'false_positives': len(false_positives),
                'false_negatives': len(false_negatives),
                'fp_rate': len(false_positives) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0,
                'fn_rate': len(false_negatives) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
            }

            self.misclassification_analysis[model_key] = analysis

            print(
                f"   Total misclassified: {total_misclassified}/{total_samples} ({analysis['misclassification_rate']:.1%})")
            print(f"   False positives: {len(false_positives)} ({analysis['fp_rate']:.1%} of negatives)")
            print(f"   False negatives: {len(false_negatives)} ({analysis['fn_rate']:.1%} of positives)")

    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations for analysis"""

        print("\n📊 CREATING COMPREHENSIVE VISUALIZATIONS")
        print("-" * 50)

        # 1. Model Performance Comparison
        self._plot_model_comparison()

        # 2. Confusion Matrices
        self._plot_confusion_matrices()

        # 3. ROC Curves (if probabilities available)
        self._plot_roc_curves()

        # 4. Misclassification Analysis
        self._plot_misclassification_analysis()

    def _plot_model_comparison(self):
        """Plot model performance comparison"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i // 2, i % 2]

            models = list(self.models.keys())
            train_scores = [self.results[m]['train_metrics'][metric] for m in models]
            val_scores = [self.results[m]['val_metrics'][metric] for m in models]
            test_scores = [self.results[m]['test_metrics'][metric] for m in models]

            x = np.arange(len(models))
            width = 0.25

            ax.bar(x - width, train_scores, width, label='Train', alpha=0.8)
            ax.bar(x, val_scores, width, label='Validation', alpha=0.8)
            ax.bar(x + width, test_scores, width, label='Test', alpha=0.8)

            ax.set_xlabel('Models')
            ax.set_ylabel(name)
            ax.set_title(f'{name} Comparison (SMOTE)')
            ax.set_xticks(x)
            ax.set_xticklabels([self.models[m]['name'].replace(' (Custom)', '') for m in models], rotation=45,
                               ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""

        n_models = len(self.models)
        cols = min(n_models, 2)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (model_key, model_data) in enumerate(self.models.items()):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue

            cm = confusion_matrix(self.y_test, model_data['test_pred'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Low Quality', 'High Quality'],
                        yticklabels=['Low Quality', 'High Quality'])
            ax.set_title(f'{model_data["name"]} - Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        # Hide unused subplots
        for i in range(n_models, len(axes)):
            if i < len(axes):
                axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def _plot_roc_curves(self):
        """Plot ROC curves if probabilities are available"""

        from sklearn.metrics import roc_curve, auc

        plt.figure(figsize=(10, 8))

        for model_key, model_data in self.models.items():
            if model_data['test_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, model_data['test_proba'])
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, linewidth=2,
                         label=f'{model_data["name"]} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison (SMOTE)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

    def _plot_misclassification_analysis(self):
        """Plot misclassification analysis"""

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        models = list(self.models.keys())
        model_names = [self.models[m]['name'].replace(' (Custom)', '') for m in models]

        # Misclassification rates
        misclass_rates = [self.misclassification_analysis[m]['misclassification_rate'] for m in models]

        bars = axes[0].bar(range(len(models)), misclass_rates, color='lightcoral', alpha=0.7)
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Misclassification Rate')
        axes[0].set_title('Misclassification Rates by Model (SMOTE)')
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)

        # Add value labels
        for bar, rate in zip(bars, misclass_rates):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                         f'{rate:.1%}', ha='center', va='bottom')

        # False positive rates
        fp_rates = [self.misclassification_analysis[m]['fp_rate'] for m in models]

        bars = axes[1].bar(range(len(models)), fp_rates, color='orange', alpha=0.7)
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('False Positive Rate')
        axes[1].set_title('False Positive Rates (SMOTE)')
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)

        for bar, rate in zip(bars, fp_rates):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                         f'{rate:.1%}', ha='center', va='bottom')

        # False negative rates
        fn_rates = [self.misclassification_analysis[m]['fn_rate'] for m in models]

        bars = axes[2].bar(range(len(models)), fn_rates, color='lightblue', alpha=0.7)
        axes[2].set_xlabel('Models')
        axes[2].set_ylabel('False Negative Rate')
        axes[2].set_title('False Negative Rates (SMOTE)')
        axes[2].set_xticks(range(len(models)))
        axes[2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)

        for bar, rate in zip(bars, fn_rates):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                         f'{rate:.1%}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def generate_final_report(self):
        """Generate final analysis report"""

        print("\n📋 FINAL ANALYSIS REPORT (SMOTE)")
        print("=" * 70)

        # Find best model
        best_model_key = max(self.results.keys(),
                             key=lambda k: self.results[k]['test_metrics']['f1'])
        best_model = self.models[best_model_key]
        best_metrics = self.results[best_model_key]['test_metrics']

        print(f"🏆 BEST PERFORMING MODEL: {best_model['name']}")
        print(f"   Test Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"   Test F1-Score: {best_metrics['f1']:.4f}")
        print(f"   Test Precision: {best_metrics['precision']:.4f}")
        print(f"   Test Recall: {best_metrics['recall']:.4f}")

        # Overfitting analysis
        print(f"\n🔍 OVERFITTING ANALYSIS:")
        for model_key, model_data in self.models.items():
            train_acc = self.results[model_key]['train_metrics']['accuracy']
            val_acc = self.results[model_key]['val_metrics']['accuracy']
            test_acc = self.results[model_key]['test_metrics']['accuracy']

            gap = train_acc - val_acc
            status = "🔴 High" if gap > 0.05 else "🟡 Moderate" if gap > 0.02 else "🟢 Low"

            print(f"   {model_data['name']}: {status} (Train: {train_acc:.3f}, Val: {val_acc:.3f}, Gap: {gap:.3f})")

        # Misclassification insights
        print(f"\n🎯 MISCLASSIFICATION INSIGHTS:")
        for model_key, analysis in self.misclassification_analysis.items():
            model_name = self.models[model_key]['name']
            fp_rate = analysis['fp_rate']
            fn_rate = analysis['fn_rate']

            print(f"   {model_name}:")
            print(f"      False Positive Rate: {fp_rate:.1%} (predicting high quality when it's low)")
            print(f"      False Negative Rate: {fn_rate:.1%} (predicting low quality when it's high)")

            if fp_rate > fn_rate:
                print(f"      → Model tends to be optimistic about wine quality")
            elif fn_rate > fp_rate:
                print(f"      → Model tends to be pessimistic about wine quality")
            else:
                print(f"      → Model shows balanced error distribution")

        print(f"\n✅ Analysis complete with SMOTE oversampling! Check visualizations above for detailed insights.")

        return {
            'best_model': best_model_key,
            'best_metrics': best_metrics,
            'timestamp': datetime.now().isoformat()
        }

    def save_complete_results(self):
        """Save all results and analysis"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare results for saving
        complete_results = {
            'timestamp': timestamp,
            'data_info': {
                'total_samples': len(self.data) if self.data is not None else 0,
                'train_samples': len(self.X_train) if self.X_train is not None else 0,
                'val_samples': len(self.X_val) if self.X_val is not None else 0,
                'test_samples': len(self.X_test) if self.X_test is not None else 0,
                'features': self.X_train.shape[1] if self.X_train is not None else 0,
                'smote_applied': True
            },
            'model_results': self.results,
            'misclassification_analysis': self.misclassification_analysis,
            'models_info': {k: {'name': v['name']} for k, v in self.models.items()}
        }

        # Save as JSON
        json_path = os.path.join(self.results_dir, f'smote_analysis_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)

        # Save as pickle (with full objects)
        pickle_path = os.path.join(self.results_dir, f'smote_analysis_{timestamp}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'results': complete_results,
                'workflow_object': self  # Save entire workflow object
            }, f)

        print(f"\n💾 RESULTS SAVED:")
        print(f"   JSON: {json_path}")
        print(f"   Pickle: {pickle_path}")

        return json_path, pickle_path

    def run_complete_analysis(self, red_path="data/winequality-red.csv",
                              white_path="data/winequality-white.csv"):
        """Run the complete analysis workflow with SMOTE"""

        try:
            # Step 1: Load and preprocess data with SMOTE
            if not self.load_and_preprocess_data(red_path, white_path):
                return False

            # Step 2: Train custom models
            if not self.train_custom_models():
                return False

            # Step 3: Analyze misclassifications
            self.analyze_misclassifications()

            # Step 4: Create visualizations
            self.create_comprehensive_visualizations()

            # Step 5: Generate final report
            final_report = self.generate_final_report()

            # Step 6: Save results
            self.save_complete_results()

            print(f"\n🎉 COMPLETE SMOTE ANALYSIS FINISHED SUCCESSFULLY!")
            return True

        except Exception as e:
            print(f"❌ Analysis failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


# Example usage
def run_smote_workflow():
    """Example of how to use the SMOTE workflow"""

    # Create workflow instance
    workflow = CompleteAnalysisWorkflow(results_dir="results")

    # Run complete analysis with SMOTE
    success = workflow.run_complete_analysis(
        red_path="data/winequality-red.csv",
        white_path="data/winequality-white.csv"
    )

    if success:
        print("🎉 SMOTE Workflow completed successfully!")
    else:
        print("❌ SMOTE Workflow failed!")

    return workflow


# Additional utility functions for loading and analyzing saved results
def load_smote_results(results_dir="results"):
    """Load SMOTE analysis results"""

    import glob

    # Find all SMOTE result files
    json_files = glob.glob(os.path.join(results_dir, "smote_analysis_*.json"))

    if not json_files:
        print("❌ No SMOTE result files found")
        return None

    # Load the most recent results
    json_files.sort(key=lambda x: os.path.getctime(x), reverse=True)
    latest_file = json_files[0]

    print("📊 LOADING SMOTE RESULTS")
    print("=" * 50)

    with open(latest_file, 'r') as f:
        data = json.load(f)

    filename = os.path.basename(latest_file)
    print(f"📁 Loaded: {filename}")
    print(f"   Timestamp: {data['timestamp']}")
    print(f"   Total samples: {data['data_info']['total_samples']}")
    print(f"   Training samples (with SMOTE): {data['data_info']['train_samples']}")

    # Display model performance
    print(f"\n🏆 MODEL PERFORMANCE:")
    print("-" * 40)

    for model_key, model_results in data['model_results'].items():
        model_name = data['models_info'][model_key]['name']
        test_metrics = model_results['test_metrics']

        print(f"\n📈 {model_name}:")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   F1-Score: {test_metrics['f1']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall: {test_metrics['recall']:.4f}")

    return data


def create_smote_comparison_plots(results_dir="results"):
    """Create comparison plots from SMOTE results"""

    data = load_smote_results(results_dir)

    if not data:
        return

    # Create performance comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['accuracy', 'f1', 'precision', 'recall']
    metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i // 2, i % 2]

        models = list(data['model_results'].keys())
        model_names = [data['models_info'][m]['name'].replace(' (Custom)', '') for m in models]

        # Get test scores
        scores = [data['model_results'][m]['test_metrics'][metric] for m in models]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']

        bars = ax.bar(range(len(models)), scores, color=colors[:len(models)], alpha=0.8)

        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Models')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison (SMOTE)')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def export_smote_results_to_csv(results_dir="results", output_file="smote_model_results.csv"):
    """Export SMOTE results to CSV for external analysis"""

    data = load_smote_results(results_dir)

    if not data:
        return

    # Prepare data for CSV
    csv_data = []

    timestamp = data['timestamp']
    experiment_id = f"smote_{timestamp}"

    # Extract model results
    for model_name, model_results in data['model_results'].items():
        model_display_name = data['models_info'][model_name]['name']

        for split in ['train_metrics', 'val_metrics', 'test_metrics']:
            if split in model_results:
                metrics = model_results[split]

                row = {
                    'experiment_id': experiment_id,
                    'timestamp': timestamp,
                    'model': model_display_name,
                    'split': split.replace('_metrics', ''),
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'smote_applied': True
                }

                # Add misclassification data if available
                if split == 'test_metrics' and 'misclassification_analysis' in data:
                    if model_name in data['misclassification_analysis']:
                        misc_data = data['misclassification_analysis'][model_name]
                        row.update({
                            'misclassification_rate': misc_data['misclassification_rate'],
                            'false_positive_rate': misc_data['fp_rate'],
                            'false_negative_rate': misc_data['fn_rate']
                        })

                csv_data.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    output_path = os.path.join(results_dir, output_file)
    df.to_csv(output_path, index=False)

    print(f"📊 SMOTE Results exported to: {output_path}")
    print(f"   Total rows: {len(df)}")
    print(f"   Models: {df['model'].nunique()}")

    return df


# Main execution
if __name__ == "__main__":
    print("🍷 Wine Quality Classification - Complete Analysis Workflow (SMOTE Only)")
    print("=" * 70)
    print("This script provides a complete workflow for:")
    print("✅ Data loading and preprocessing with SMOTE")
    print("✅ Custom model training and evaluation")
    print("✅ Comprehensive performance analysis")
    print("✅ Misclassification pattern analysis")
    print("✅ Detailed visualizations")
    print("✅ Results saving and comparison")
    print("\n📋 Usage Examples:")
    print("=" * 30)
    print("# Run complete SMOTE analysis:")
    print("workflow = CompleteAnalysisWorkflow()")
    print("workflow.run_complete_analysis()")
    print("")
    print("# Load and analyze saved results:")
    print("load_smote_results('results')")
    print("")
    print("# Create comparison plots:")
    print("create_smote_comparison_plots('results')")
    print("")
    print("# Export to CSV:")
    print("export_smote_results_to_csv('results')")
    print("\n🚀 Ready to run SMOTE analysis!")


# Notebook helper class for SMOTE only
class SMOTEAnalysisHelper:
    """Helper class for SMOTE analysis in Jupyter notebooks"""

    @staticmethod
    def quick_smote_analysis(red_path="data/winequality-red.csv",
                             white_path="data/winequality-white.csv"):
        """Quick SMOTE analysis for notebook environments"""

        workflow = CompleteAnalysisWorkflow()
        success = workflow.run_complete_analysis(red_path, white_path)

        if success:
            print("\n✅ Quick SMOTE analysis completed!")
            return workflow
        else:
            print("\n❌ Quick SMOTE analysis failed!")
            return None

    @staticmethod
    def analyze_existing_smote_results(results_dir="results"):
        """Analyze existing SMOTE results"""

        results = load_smote_results(results_dir)
        if results:
            create_smote_comparison_plots(results_dir)
            export_smote_results_to_csv(results_dir)

        return results


print("\n🎯 All SMOTE workflow components ready!")
print("Use CompleteAnalysisWorkflow() to start your SMOTE analysis!")