"""
Overfitting and Underfitting Analysis Module - FIXED VERSION

This module provides comprehensive analysis to detect and visualize overfitting and underfitting
in machine learning models. It analyzes training vs validation performance, learning curves,
and provides recommendations for model improvement.

FIXES:
- Proper directory creation and path handling
- Fixed saving mechanism to ensure plots are saved before showing
- Added verbose logging for debugging save operations
- Improved error handling for file operations

Usage:
    from src.overfitting_analysis import OverfittingAnalyzer

    analyzer = OverfittingAnalyzer()
    analyzer.analyze_all_models(models_dict, X_train, y_train, X_test, y_test)
    analyzer.create_comprehensive_analysis(save_plots=True)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
import warnings
import os

warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")


class OverfittingAnalyzer:
    """
    Comprehensive overfitting and underfitting analysis for machine learning models
    """

    def __init__(self):
        self.models = {}
        self.analysis_results = {}
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def analyze_all_models(self, models_dict, X_train, y_train, X_test, y_test, model_names=None):
        """
        Analyze overfitting/underfitting for all models

        Args:
            models_dict (dict): Dictionary of trained models
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_names (dict): Optional display names for models
        """
        print("\n🔍 OVERFITTING/UNDERFITTING ANALYSIS")
        print("=" * 60)

        # Store data
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

        # Add models
        for model_key, model in models_dict.items():
            model_name = model_names.get(model_key) if model_names else None
            self.add_model_analysis(model_key, model, model_name)

        # Analyze each model
        for model_key in self.models.keys():
            print(f"\n📊 Analyzing {self.models[model_key]['name']}...")
            self._analyze_single_model(model_key)

        print("\n✅ Overfitting/Underfitting analysis complete!")

    def add_model_analysis(self, model_key, model, model_name=None):
        """Add a model for analysis"""
        self.models[model_key] = {
            'model': model,
            'name': model_name or model_key.replace('_', ' ').title()
        }

    def _analyze_single_model(self, model_key):
        """Analyze overfitting/underfitting for a single model"""
        model = self.models[model_key]['model']
        model_name = self.models[model_key]['name']

        # 1. Basic train vs test performance comparison
        train_performance = self._get_model_performance(model, self.X_train, self.y_train)
        test_performance = self._get_model_performance(model, self.X_test, self.y_test)

        # 2. Analyze training curves if available
        training_curve_analysis = self._analyze_training_curves(model)

        # 3. Generate learning curves
        learning_curve_analysis = self._generate_learning_curves(model_key, model)

        # 4. Analyze model complexity vs performance
        complexity_analysis = self._analyze_model_complexity(model_key, model)

        # 5. Detect overfitting/underfitting patterns
        fitting_diagnosis = self._diagnose_fitting_issues(
            train_performance, test_performance, training_curve_analysis,
            learning_curve_analysis, complexity_analysis
        )

        # Store results
        self.analysis_results[model_key] = {
            'model_name': model_name,
            'train_performance': train_performance,
            'test_performance': test_performance,
            'training_curves': training_curve_analysis,
            'learning_curves': learning_curve_analysis,
            'complexity_analysis': complexity_analysis,
            'fitting_diagnosis': fitting_diagnosis
        }

        # Print summary
        self._print_model_summary(model_key)

    def _get_model_performance(self, model, X, y):
        """Get comprehensive performance metrics for a model"""
        try:
            predictions = model.predict(X)
            accuracy = accuracy_score(y, predictions)
            f1 = f1_score(y, predictions, average='weighted')

            # Try to get prediction probabilities for additional metrics
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
                elif hasattr(model, 'decision_function'):
                    scores = model.decision_function(X)
                    probabilities = 1 / (1 + np.exp(-np.array(scores)))
                else:
                    probabilities = None
            except:
                probabilities = None

            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': predictions,
                'probabilities': probabilities,
                'n_samples': len(y)
            }
        except Exception as e:
            print(f"   ⚠️ Error getting performance: {e}")
            return {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'predictions': None,
                'probabilities': None,
                'n_samples': len(y),
                'error': str(e)
            }

    def _analyze_training_curves(self, model):
        """Analyze training loss curves if available"""
        if hasattr(model, 'losses') and len(model.losses) > 0:
            losses = np.array(model.losses)

            # Analyze convergence
            if len(losses) > 20:
                # Check if loss is still decreasing at the end
                recent_trend = np.polyfit(range(len(losses) // 2, len(losses)),
                                          losses[len(losses) // 2:], 1)[0]
                converged = abs(recent_trend) < 1e-6

                # Check for oscillations
                if len(losses) > 50:
                    recent_losses = losses[-20:]
                    oscillation_variance = np.var(recent_losses)
                else:
                    oscillation_variance = 0

                # Early vs late loss comparison
                early_loss = np.mean(losses[:len(losses) // 4]) if len(losses) > 4 else losses[0]
                late_loss = np.mean(losses[-len(losses) // 4:]) if len(losses) > 4 else losses[-1]
                improvement_ratio = (early_loss - late_loss) / early_loss if early_loss > 0 else 0

                return {
                    'available': True,
                    'losses': losses,
                    'converged': converged,
                    'final_loss': losses[-1],
                    'best_loss': np.min(losses),
                    'improvement_ratio': improvement_ratio,
                    'oscillation_variance': oscillation_variance,
                    'trend_slope': recent_trend,
                    'epochs_trained': len(losses)
                }
            else:
                return {
                    'available': True,
                    'losses': losses,
                    'converged': False,
                    'final_loss': losses[-1] if len(losses) > 0 else 0,
                    'epochs_trained': len(losses),
                    'insufficient_data': True
                }
        else:
            return {'available': False}

    def _generate_learning_curves(self, model_key, model):
        """Generate learning curves to analyze overfitting"""
        try:
            print(f"     📈 Generating learning curves...")

            # Create a wrapper for models that need special handling
            model_wrapper = self._create_model_wrapper(model_key, model)

            # Generate learning curve data
            train_sizes = np.linspace(0.1, 1.0, 10)

            train_scores_all = []
            val_scores_all = []
            actual_train_sizes = []

            # Manual learning curve generation for better control
            n_samples = len(self.X_train)

            for train_size in train_sizes:
                n_train = int(train_size * n_samples)
                if n_train < 10:  # Skip very small sample sizes
                    continue

                # Sample training data
                indices = np.random.choice(n_samples, n_train, replace=False)
                X_subset = self.X_train[indices]
                y_subset = self.y_train[indices]

                # Train model on subset
                try:
                    subset_model = self._create_fresh_model(model_key, model)
                    if hasattr(subset_model, 'fit'):
                        if 'svm' in model_key.lower() and hasattr(subset_model, 'fit'):
                            # SVM models need max_iter parameter
                            subset_model.fit(X_subset, y_subset, max_iter=min(1000, len(y_subset) * 2))
                        else:
                            subset_model.fit(X_subset, y_subset)

                    # Evaluate on training subset and validation set
                    train_pred = subset_model.predict(X_subset)
                    val_pred = subset_model.predict(self.X_test)

                    train_acc = accuracy_score(y_subset, train_pred)
                    val_acc = accuracy_score(self.y_test, val_pred)

                    train_scores_all.append(train_acc)
                    val_scores_all.append(val_acc)
                    actual_train_sizes.append(n_train)

                except Exception as e:
                    print(f"       ⚠️ Error at train_size {train_size}: {e}")
                    continue

            if len(train_scores_all) >= 3:
                return {
                    'available': True,
                    'train_sizes': np.array(actual_train_sizes),
                    'train_scores': np.array(train_scores_all),
                    'val_scores': np.array(val_scores_all),
                    'gap_at_end': train_scores_all[-1] - val_scores_all[-1] if len(train_scores_all) > 0 else 0
                }
            else:
                return {'available': False, 'error': 'Insufficient learning curve data'}

        except Exception as e:
            print(f"     ⚠️ Learning curve generation failed: {e}")
            return {'available': False, 'error': str(e)}

    def _create_model_wrapper(self, model_key, original_model):
        """Create a wrapper for models that work with sklearn learning_curve"""

        class ModelWrapper:
            def __init__(self, original_model, model_key):
                self.original_model = original_model
                self.model_key = model_key

            def fit(self, X, y):
                if 'svm' in self.model_key.lower():
                    self.original_model.fit(X, y, max_iter=1000)
                else:
                    self.original_model.fit(X, y)
                return self

            def predict(self, X):
                return self.original_model.predict(X)

            def score(self, X, y):
                predictions = self.predict(X)
                return accuracy_score(y, predictions)

        return ModelWrapper(original_model, model_key)

    def _create_fresh_model(self, model_key, original_model):
        """Create a fresh instance of the model with same parameters"""
        if 'lr' in model_key.lower():
            from models.logistic.base_logistic import LogisticRegressionScratch
            return LogisticRegressionScratch(
                learning_rate=getattr(original_model, 'learning_rate', 0.1),
                regularization_strength=getattr(original_model, 'lambda_', 0.01),
                epochs=getattr(original_model, 'epochs', 1000)
            )
        elif 'svm' in model_key.lower() and 'kernel' not in model_key.lower():
            from models.svm.base_svm import SVMClassifierScratch
            return SVMClassifierScratch(
                lambda_=getattr(original_model, 'lambda_', 0.01)
            )
        elif 'klr' in model_key.lower():
            from models.logistic.kernel_logistic import KernelLogisticRegression
            return KernelLogisticRegression(
                kernel=getattr(original_model, 'kernel', None),
                lambda_=getattr(original_model, 'lambda_', 0.01),
                epochs=min(500, getattr(original_model, 'epochs', 500)),  # Reduce epochs for speed
                subsample_ratio=0.1,  # Use smaller subset for learning curves
                batch_size=32
            )
        elif 'ksvm' in model_key.lower():
            from models.svm.base_svm import KernelPegasosSVM
            return KernelPegasosSVM(
                kernel=getattr(original_model, 'kernel', None),
                lambda_=getattr(original_model, 'lambda_', 0.01),
                max_iter=min(500, getattr(original_model, 'max_iter', 500))  # Reduce iterations for speed
            )
        else:
            # Fallback - try to copy the original model
            return original_model

    def _analyze_model_complexity(self, model_key, model):
        """Analyze model complexity indicators"""
        complexity_indicators = {}

        # Model-specific complexity analysis
        if 'lr' in model_key.lower():
            # Logistic Regression: weight magnitude, regularization
            if hasattr(model, 'weights'):
                weights = np.array(model.weights)
                complexity_indicators.update({
                    'weight_magnitude': np.linalg.norm(weights),
                    'max_weight': np.max(np.abs(weights)),
                    'weight_sparsity': np.sum(np.abs(weights) < 0.01) / len(weights),
                    'regularization': getattr(model, 'lambda_', 0)
                })

        elif 'svm' in model_key.lower() and 'kernel' not in model_key.lower():
            # Linear SVM: weight magnitude
            if hasattr(model, 'weights'):
                weights = np.array(model.weights)
                complexity_indicators.update({
                    'weight_magnitude': np.linalg.norm(weights),
                    'regularization': getattr(model, 'lambda_', 0)
                })

        elif 'kernel' in model_key.lower():
            # Kernel methods: number of support vectors
            if hasattr(model, 'support_vectors'):
                n_support = len(model.support_vectors)
                complexity_indicators.update({
                    'n_support_vectors': n_support,
                    'support_vector_ratio': n_support / len(self.X_train),
                    'regularization': getattr(model, 'lambda_', 0)
                })
            elif hasattr(model, 'X_support'):
                n_support = len(model.X_support) if model.X_support is not None else 0
                complexity_indicators.update({
                    'n_support_vectors': n_support,
                    'support_vector_ratio': n_support / len(self.X_train) if len(self.X_train) > 0 else 0,
                    'regularization': getattr(model, 'lambda_', 0)
                })

        # General complexity indicators
        complexity_indicators.update({
            'training_samples': len(self.X_train),
            'features': self.X_train.shape[1],
            'samples_to_features_ratio': len(self.X_train) / self.X_train.shape[1]
        })

        return complexity_indicators

    def _diagnose_fitting_issues(self, train_perf, test_perf, training_curves,
                                 learning_curves, complexity_analysis):
        """Diagnose overfitting, underfitting, or good fit"""

        diagnosis = {
            'fitting_status': 'unknown',
            'confidence': 0.0,
            'indicators': [],
            'recommendations': []
        }

        # Calculate performance gap
        performance_gap = train_perf['accuracy'] - test_perf['accuracy']

        # Collect evidence
        overfitting_evidence = 0
        underfitting_evidence = 0
        good_fit_evidence = 0

        # Evidence from performance gap
        if performance_gap > 0.1:
            overfitting_evidence += 3
            diagnosis['indicators'].append(f"Large train-test gap ({performance_gap:.3f})")
        elif performance_gap > 0.05:
            overfitting_evidence += 1
            diagnosis['indicators'].append(f"Moderate train-test gap ({performance_gap:.3f})")
        elif performance_gap < 0.02:
            good_fit_evidence += 2
            diagnosis['indicators'].append(f"Small train-test gap ({performance_gap:.3f})")

        # Evidence from absolute performance
        if test_perf['accuracy'] < 0.6:
            underfitting_evidence += 2
            diagnosis['indicators'].append(f"Low test accuracy ({test_perf['accuracy']:.3f})")
        elif test_perf['accuracy'] > 0.8:
            good_fit_evidence += 1
            diagnosis['indicators'].append(f"Good test accuracy ({test_perf['accuracy']:.3f})")

        # Evidence from training curves
        if training_curves['available']:
            if not training_curves.get('converged', True):
                underfitting_evidence += 1
                diagnosis['indicators'].append("Training loss not converged")

            improvement = training_curves.get('improvement_ratio', 0)
            if improvement < 0.1:
                underfitting_evidence += 1
                diagnosis['indicators'].append(f"Limited training improvement ({improvement:.3f})")
            elif improvement > 0.8:
                good_fit_evidence += 1
                diagnosis['indicators'].append(f"Good training improvement ({improvement:.3f})")

        # Evidence from learning curves
        if learning_curves['available']:
            final_gap = learning_curves.get('gap_at_end', 0)
            if final_gap > 0.1:
                overfitting_evidence += 2
                diagnosis['indicators'].append(f"Learning curve shows overfitting (gap: {final_gap:.3f})")
            elif final_gap < 0.03:
                good_fit_evidence += 1
                diagnosis['indicators'].append(f"Learning curve shows good fit (gap: {final_gap:.3f})")

        # Evidence from model complexity
        samples_to_features = complexity_analysis.get('samples_to_features_ratio', 1)
        if samples_to_features < 10:
            overfitting_evidence += 1
            diagnosis['indicators'].append(f"High complexity vs data size ratio ({samples_to_features:.1f})")
        elif samples_to_features > 50:
            good_fit_evidence += 1
            diagnosis['indicators'].append(f"Good data size vs complexity ratio ({samples_to_features:.1f})")

        # Make diagnosis
        total_evidence = overfitting_evidence + underfitting_evidence + good_fit_evidence

        if total_evidence == 0:
            diagnosis['fitting_status'] = 'inconclusive'
            diagnosis['confidence'] = 0.0
        else:
            overfitting_score = overfitting_evidence / total_evidence
            underfitting_score = underfitting_evidence / total_evidence
            good_fit_score = good_fit_evidence / total_evidence

            max_score = max(overfitting_score, underfitting_score, good_fit_score)
            diagnosis['confidence'] = max_score

            if overfitting_score == max_score:
                diagnosis['fitting_status'] = 'overfitting'
            elif underfitting_score == max_score:
                diagnosis['fitting_status'] = 'underfitting'
            else:
                diagnosis['fitting_status'] = 'good_fit'

        # Generate recommendations
        if diagnosis['fitting_status'] == 'overfitting':
            diagnosis['recommendations'].extend([
                "Increase regularization strength",
                "Reduce model complexity",
                "Collect more training data",
                "Use dropout or early stopping",
                "Try simpler model architecture"
            ])
        elif diagnosis['fitting_status'] == 'underfitting':
            diagnosis['recommendations'].extend([
                "Decrease regularization strength",
                "Increase model complexity",
                "Add more features",
                "Train for more epochs",
                "Try more complex model architecture"
            ])
        elif diagnosis['fitting_status'] == 'good_fit':
            diagnosis['recommendations'].extend([
                "Model appears well-fitted",
                "Consider ensemble methods for further improvement",
                "Validate on additional test sets"
            ])

        return diagnosis

    def _print_model_summary(self, model_key):
        """Print summary of fitting analysis for a model"""
        analysis = self.analysis_results[model_key]
        model_name = analysis['model_name']
        diagnosis = analysis['fitting_diagnosis']

        print(f"   📈 {model_name} Fitting Analysis:")
        print(f"      Train Accuracy: {analysis['train_performance']['accuracy']:.4f}")
        print(f"      Test Accuracy:  {analysis['test_performance']['accuracy']:.4f}")
        print(
            f"      Performance Gap: {analysis['train_performance']['accuracy'] - analysis['test_performance']['accuracy']:.4f}")

        status_emoji = {
            'overfitting': '🔴',
            'underfitting': '🟡',
            'good_fit': '🟢',
            'inconclusive': '⚪'
        }

        status = diagnosis['fitting_status']
        confidence = diagnosis['confidence']
        emoji = status_emoji.get(status, '⚪')

        print(f"      Diagnosis: {emoji} {status.upper()} (confidence: {confidence:.2f})")

        if diagnosis['indicators']:
            print(f"      Key indicators:")
            for indicator in diagnosis['indicators'][:3]:  # Show top 3
                print(f"         • {indicator}")

    def _ensure_directory_exists(self, directory_path):
        """Ensure directory exists and log the operation"""
        try:
            os.makedirs(directory_path, exist_ok=True)
            print(f"   📁 Directory ensured: {os.path.abspath(directory_path)}")
            return True
        except Exception as e:
            print(f"   ❌ Error creating directory {directory_path}: {e}")
            return False

    def _save_plot_safely(self, figure, filepath, dpi=300):
        """Save plot with proper error handling and logging"""
        try:
            # Ensure the directory exists
            directory = os.path.dirname(filepath)
            if not self._ensure_directory_exists(directory):
                return False

            # Get absolute path for logging
            abs_filepath = os.path.abspath(filepath)

            # Save the figure
            figure.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')

            # Verify file was created
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"   ✅ Plot saved successfully: {abs_filepath} ({file_size} bytes)")
                return True
            else:
                print(f"   ❌ Plot file was not created: {abs_filepath}")
                return False

        except Exception as e:
            print(f"   ❌ Error saving plot to {filepath}: {e}")
            return False

    def plot_fitting_analysis(self, figsize=(20, 15), save_plots=False, save_dir='output/evaluation_plots'):
        """Create comprehensive fitting analysis visualizations with fixed saving"""

        print("📊 Creating Overfitting/Underfitting Analysis Plots...")

        if not self.analysis_results:
            print("❌ No analysis data available")
            return

        # Create save directory if needed and log it
        if save_plots:
            if not self._ensure_directory_exists(save_dir):
                print("❌ Failed to create save directory, disabling save_plots")
                save_plots = False

        n_models = len(self.analysis_results)

        # Create a large subplot grid
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Plot 1: Train vs Test Performance
        ax1 = fig.add_subplot(gs[0, 0])

        model_names = []
        train_accs = []
        test_accs = []
        gaps = []

        for model_key, analysis in self.analysis_results.items():
            model_names.append(analysis['model_name'].replace(' (Custom)', ''))
            train_acc = analysis['train_performance']['accuracy']
            test_acc = analysis['test_performance']['accuracy']
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            gaps.append(train_acc - test_acc)

        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, train_accs, width, label='Train', alpha=0.8, color='lightblue')
        bars2 = ax1.bar(x + width / 2, test_accs, width, label='Test', alpha=0.8, color='lightcoral')

        ax1.set_ylabel('Accuracy')
        ax1.set_title('Train vs Test Performance', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Add gap annotations
        for i, gap in enumerate(gaps):
            ax1.annotate(f'{gap:.3f}', xy=(i, max(train_accs[i], test_accs[i]) + 0.02),
                         ha='center', va='bottom', fontsize=8, fontweight='bold',
                         color='red' if gap > 0.05 else 'green')

        # Plot 2: Performance Gaps
        ax2 = fig.add_subplot(gs[0, 1])

        colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' for gap in gaps]
        bars = ax2.bar(range(len(model_names)), gaps, color=colors, alpha=0.7)

        ax2.set_ylabel('Train - Test Gap')
        ax2.set_title('Overfitting Indicator\n(Performance Gap)', fontweight='bold')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High threshold')
        ax2.legend()

        # Add value labels
        for bar, gap in zip(bars, gaps):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{gap:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 3: Fitting Status Distribution
        ax3 = fig.add_subplot(gs[0, 2])

        status_counts = {}
        for analysis in self.analysis_results.values():
            status = analysis['fitting_diagnosis']['fitting_status']
            status_counts[status] = status_counts.get(status, 0) + 1

        statuses = list(status_counts.keys())
        counts = list(status_counts.values())
        colors_map = {'overfitting': 'red', 'underfitting': 'orange', 'good_fit': 'green', 'inconclusive': 'gray'}
        pie_colors = [colors_map.get(status, 'gray') for status in statuses]

        wedges, texts, autotexts = ax3.pie(counts, labels=statuses, colors=pie_colors, autopct='%1.0f%%',
                                           startangle=90, textprops={'fontsize': 10})
        ax3.set_title('Fitting Status Distribution', fontweight='bold')

        # Plot 4: Model Complexity Analysis
        ax4 = fig.add_subplot(gs[0, 3])

        # Extract complexity metrics where available
        complexity_metrics = []
        model_names_complex = []

        for model_key, analysis in self.analysis_results.items():
            complexity = analysis['complexity_analysis']
            model_names_complex.append(analysis['model_name'].replace(' (Custom)', ''))

            # Use different complexity measures for different models
            if 'n_support_vectors' in complexity:
                complexity_metrics.append(complexity['n_support_vectors'])
            elif 'weight_magnitude' in complexity:
                complexity_metrics.append(complexity['weight_magnitude'] * 100)  # Scale for visibility
            else:
                complexity_metrics.append(complexity.get('samples_to_features_ratio', 1))

        if complexity_metrics:
            bars = ax4.bar(range(len(model_names_complex)), complexity_metrics,
                           color='mediumpurple', alpha=0.7)
            ax4.set_ylabel('Complexity Measure')
            ax4.set_title('Model Complexity', fontweight='bold')
            ax4.set_xticks(range(len(model_names_complex)))
            ax4.set_xticklabels(model_names_complex, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)

            # Add value labels
            for bar, metric in zip(bars, complexity_metrics):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height + max(complexity_metrics) * 0.01,
                         f'{metric:.1f}', ha='center', va='bottom', fontsize=8)

        # Plot 5-8: Individual Learning Curves
        learning_curve_plots = 0
        for i, (model_key, analysis) in enumerate(self.analysis_results.items()):
            if learning_curve_plots >= 4:
                break

            row = 1 + learning_curve_plots // 2
            col = learning_curve_plots % 2
            ax = fig.add_subplot(gs[row, col])

            learning_curves = analysis['learning_curves']
            if learning_curves['available']:
                train_sizes = learning_curves['train_sizes']
                train_scores = learning_curves['train_scores']
                val_scores = learning_curves['val_scores']

                ax.plot(train_sizes, train_scores, 'o-', color='blue', alpha=0.8,
                        label='Training', linewidth=2, markersize=4)
                ax.plot(train_sizes, val_scores, 'o-', color='red', alpha=0.8,
                        label='Validation', linewidth=2, markersize=4)

                ax.fill_between(train_sizes, train_scores, val_scores, alpha=0.1,
                                color='red' if train_scores[-1] > val_scores[-1] + 0.05 else 'green')

                ax.set_xlabel('Training Set Size')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'{analysis["model_name"].replace(" (Custom)", "")}\nLearning Curve',
                             fontweight='bold', fontsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)

                # Add gap annotation
                if len(train_scores) > 0 and len(val_scores) > 0:
                    final_gap = train_scores[-1] - val_scores[-1]
                    ax.text(0.02, 0.98, f'Final Gap: {final_gap:.3f}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontsize=8)
            else:
                ax.text(0.5, 0.5, 'Learning curve\nnot available',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, style='italic')
                ax.set_title(f'{analysis["model_name"].replace(" (Custom)", "")}',
                             fontweight='bold', fontsize=10)

            learning_curve_plots += 1

        # Plot 9: Training Loss Curves (bottom row)
        ax9 = fig.add_subplot(gs[2, :2])

        loss_curves_available = False
        for model_key, analysis in self.analysis_results.items():
            training_curves = analysis['training_curves']
            if training_curves['available'] and not training_curves.get('insufficient_data', False):
                losses = training_curves['losses']
                epochs = range(1, len(losses) + 1)

                model_name = analysis['model_name'].replace(' (Custom)', '')
                ax9.plot(epochs, losses, label=model_name, linewidth=2, alpha=0.8)
                loss_curves_available = True

        if loss_curves_available:
            ax9.set_xlabel('Epoch')
            ax9.set_ylabel('Training Loss')
            ax9.set_title('Training Loss Convergence', fontweight='bold')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            ax9.set_yscale('log')  # Log scale for better visualization
        else:
            ax9.text(0.5, 0.5, 'No training loss data available',
                     ha='center', va='center', transform=ax9.transAxes,
                     fontsize=14, style='italic')
            ax9.set_title('Training Loss Convergence', fontweight='bold')

        # Plot 10: Recommendations Summary
        ax10 = fig.add_subplot(gs[2, 2:])
        ax10.axis('off')

        # Create recommendations text
        recommendations_text = "📋 FITTING ANALYSIS RECOMMENDATIONS\n" + "=" * 50 + "\n\n"

        for model_key, analysis in self.analysis_results.items():
            model_name = analysis['model_name']
            diagnosis = analysis['fitting_diagnosis']
            status = diagnosis['fitting_status']
            confidence = diagnosis['confidence']

            # Status emoji
            status_emoji = {
                'overfitting': '🔴 OVERFITTING',
                'underfitting': '🟡 UNDERFITTING',
                'good_fit': '🟢 GOOD FIT',
                'inconclusive': '⚪ INCONCLUSIVE'
            }

            recommendations_text += f"{status_emoji.get(status, '⚪ UNKNOWN')} - {model_name}\n"
            recommendations_text += f"   Confidence: {confidence:.2f}\n"

            if diagnosis['recommendations']:
                recommendations_text += f"   Recommendations:\n"
                for rec in diagnosis['recommendations'][:3]:  # Top 3 recommendations
                    recommendations_text += f"   • {rec}\n"
            recommendations_text += "\n"

        ax10.text(0.05, 0.95, recommendations_text, transform=ax10.transAxes,
                  verticalalignment='top', fontfamily='monospace', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.suptitle('Comprehensive Overfitting/Underfitting Analysis',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        # Save plot BEFORE showing it if requested
        if save_plots:
            filename = os.path.join(save_dir, 'overfitting_analysis_comprehensive.png')
            if self._save_plot_safely(fig, filename):
                print(f"   💾 Comprehensive analysis plot saved successfully")
            else:
                print(f"   ❌ Failed to save comprehensive analysis plot")

        plt.show()

    def plot_detailed_learning_curves(self, figsize=(16, 12), save_plots=False, save_dir='output/evaluation_plots'):
        """Create detailed learning curves for all models with fixed saving"""

        print("📈 Creating Detailed Learning Curves...")

        if save_plots:
            if not self._ensure_directory_exists(save_dir):
                print("❌ Failed to create save directory, disabling save_plots")
                save_plots = False

        # Filter models with learning curve data
        models_with_curves = {
            k: v for k, v in self.analysis_results.items()
            if v['learning_curves']['available']
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

        for i, (model_key, analysis) in enumerate(models_with_curves.items()):
            if i >= len(axes):
                break

            ax = axes[i]
            learning_curves = analysis['learning_curves']

            train_sizes = learning_curves['train_sizes']
            train_scores = learning_curves['train_scores']
            val_scores = learning_curves['val_scores']

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
            model_name = analysis['model_name'].replace(' (Custom)', '')
            ax.set_xlabel('Training Set Size', fontweight='bold')
            ax.set_ylabel('Accuracy Score', fontweight='bold')
            ax.set_title(f'{model_name} Learning Curve', fontweight='bold', fontsize=14)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

            # Add annotations
            if len(train_scores) > 0 and len(val_scores) > 0:
                final_gap = train_scores[-1] - val_scores[-1]
                max_gap = np.max(train_scores - val_scores)

                # Gap information
                gap_color = 'red' if final_gap > 0.1 else 'orange' if final_gap > 0.05 else 'green'
                ax.text(0.02, 0.98, f'Final Gap: {final_gap:.3f}\nMax Gap: {max_gap:.3f}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=gap_color, alpha=0.3),
                        fontsize=10, fontweight='bold')

                # Convergence information
                train_trend = np.polyfit(train_sizes[-3:], train_scores[-3:], 1)[0] if len(train_sizes) >= 3 else 0
                val_trend = np.polyfit(train_sizes[-3:], val_scores[-3:], 1)[0] if len(train_sizes) >= 3 else 0

                convergence_status = "Converging" if abs(train_trend) < 0.001 and abs(
                    val_trend) < 0.001 else "Still Learning"
                ax.text(0.02, 0.02, f'Status: {convergence_status}',
                        transform=ax.transAxes, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                        fontsize=9)

        # Hide unused subplots
        for i in range(len(models_with_curves), len(axes)):
            if i < len(axes):
                axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle('Detailed Learning Curves Analysis', fontsize=16, fontweight='bold', y=1.02)

        # Save plot BEFORE showing it if requested
        if save_plots:
            filename = os.path.join(save_dir, 'learning_curves_detailed.png')
            if self._save_plot_safely(fig, filename):
                print(f"   💾 Detailed learning curves plot saved successfully")
            else:
                print(f"   ❌ Failed to save detailed learning curves plot")

        plt.show()

    def generate_comprehensive_report(self):
        """Generate detailed text report of overfitting/underfitting analysis"""

        print("\n📋 COMPREHENSIVE OVERFITTING/UNDERFITTING ANALYSIS REPORT")
        print("=" * 80)

        if not self.analysis_results:
            print("❌ No analysis data available")
            return

        # Overall statistics
        print("\n🔍 OVERALL FINDINGS:")
        print("-" * 30)

        total_models = len(self.analysis_results)

        # Count fitting statuses
        status_counts = {}
        avg_gaps = []

        for analysis in self.analysis_results.values():
            status = analysis['fitting_diagnosis']['fitting_status']
            status_counts[status] = status_counts.get(status, 0) + 1

            gap = (analysis['train_performance']['accuracy'] -
                   analysis['test_performance']['accuracy'])
            avg_gaps.append(gap)

        avg_gap = np.mean(avg_gaps)

        print(f"   Models analyzed: {total_models}")
        print(f"   Average train-test gap: {avg_gap:.3f}")
        print(f"   Status distribution:")
        for status, count in status_counts.items():
            percentage = (count / total_models) * 100
            emoji = {'overfitting': '🔴', 'underfitting': '🟡', 'good_fit': '🟢', 'inconclusive': '⚪'}.get(status, '⚪')
            print(f"      {emoji} {status.replace('_', ' ').title()}: {count}/{total_models} ({percentage:.1f}%)")

        # Model-specific analysis
        print(f"\n📊 MODEL-SPECIFIC ANALYSIS:")
        print("-" * 35)

        for model_key, analysis in self.analysis_results.items():
            model_name = analysis['model_name']
            train_acc = analysis['train_performance']['accuracy']
            test_acc = analysis['test_performance']['accuracy']
            gap = train_acc - test_acc
            diagnosis = analysis['fitting_diagnosis']

            print(f"\n🔹 {model_name}:")
            print(f"   Performance: Train={train_acc:.4f}, Test={test_acc:.4f}, Gap={gap:.4f}")

            # Status with confidence
            status = diagnosis['fitting_status']
            confidence = diagnosis['confidence']
            status_emoji = {'overfitting': '🔴', 'underfitting': '🟡', 'good_fit': '🟢', 'inconclusive': '⚪'}.get(status,
                                                                                                               '⚪')
            print(f"   Status: {status_emoji} {status.upper()} (confidence: {confidence:.2f})")

            # Key indicators
            if diagnosis['indicators']:
                print(f"   Key indicators:")
                for indicator in diagnosis['indicators']:
                    print(f"      • {indicator}")

            # Training curve analysis
            training_curves = analysis['training_curves']
            if training_curves['available']:
                if training_curves.get('converged', False):
                    print(f"   Training: ✅ Converged after {training_curves['epochs_trained']} epochs")
                else:
                    print(f"   Training: ⚠️ May need more epochs ({training_curves['epochs_trained']} completed)")

                improvement = training_curves.get('improvement_ratio', 0)
                print(f"   Learning progress: {improvement:.1%} improvement from start to finish")

            # Learning curve analysis
            learning_curves = analysis['learning_curves']
            if learning_curves['available']:
                final_gap = learning_curves.get('gap_at_end', 0)
                if final_gap > 0.1:
                    print(f"   Learning curve: 🔴 Shows overfitting (final gap: {final_gap:.3f})")
                elif final_gap > 0.05:
                    print(f"   Learning curve: 🟡 Shows moderate overfitting (final gap: {final_gap:.3f})")
                else:
                    print(f"   Learning curve: 🟢 Shows good generalization (final gap: {final_gap:.3f})")

            # Complexity analysis
            complexity = analysis['complexity_analysis']
            samples_to_features = complexity.get('samples_to_features_ratio', 1)
            print(f"   Data complexity: {samples_to_features:.1f} samples per feature")

            if 'n_support_vectors' in complexity:
                sv_ratio = complexity['support_vector_ratio']
                print(
                    f"   Model complexity: {complexity['n_support_vectors']} support vectors ({sv_ratio:.1%} of training data)")
            elif 'weight_magnitude' in complexity:
                print(f"   Model complexity: Weight magnitude = {complexity['weight_magnitude']:.3f}")

        # Recommendations section
        print(f"\n💡 RECOMMENDATIONS BY FITTING STATUS:")
        print("-" * 45)

        # Group models by status
        status_groups = {}
        for model_key, analysis in self.analysis_results.items():
            status = analysis['fitting_diagnosis']['fitting_status']
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(analysis['model_name'])

        for status, models in status_groups.items():
            status_emoji = {'overfitting': '🔴', 'underfitting': '🟡', 'good_fit': '🟢', 'inconclusive': '⚪'}.get(status,
                                                                                                               '⚪')
            print(f"\n{status_emoji} {status.upper()} Models: {', '.join(models)}")

            if status == 'overfitting':
                print("   Recommended actions:")
                print("   • Increase regularization strength (λ)")
                print("   • Reduce model complexity")
                print("   • Collect more training data")
                print("   • Use cross-validation for model selection")
                print("   • Apply early stopping during training")
                print("   • Consider ensemble methods to reduce variance")

            elif status == 'underfitting':
                print("   Recommended actions:")
                print("   • Decrease regularization strength")
                print("   • Increase model complexity")
                print("   • Add more features or feature engineering")
                print("   • Train for more epochs/iterations")
                print("   • Try different model architectures")
                print("   • Check for data quality issues")

            elif status == 'good_fit':
                print("   Recommended actions:")
                print("   • Model appears well-fitted - good job!")
                print("   • Consider ensemble methods for marginal improvements")
                print("   • Validate performance on additional test sets")
                print("   • Monitor performance in production")

            elif status == 'inconclusive':
                print("   Recommended actions:")
                print("   • Collect more training data for clearer analysis")
                print("   • Run longer training sessions")
                print("   • Use more sophisticated validation techniques")

        # General recommendations
        print(f"\n🎯 GENERAL RECOMMENDATIONS:")
        print("-" * 30)

        high_gap_models = [
            analysis['model_name'] for analysis in self.analysis_results.values()
            if (analysis['train_performance']['accuracy'] - analysis['test_performance']['accuracy']) > 0.1
        ]

        low_performance_models = [
            analysis['model_name'] for analysis in self.analysis_results.values()
            if analysis['test_performance']['accuracy'] < 0.7
        ]

        if high_gap_models:
            print(f"   📊 Models with high train-test gaps: {', '.join(high_gap_models)}")
            print("      → Focus on regularization and data collection")

        if low_performance_models:
            print(f"   📈 Models with low overall performance: {', '.join(low_performance_models)}")
            print("      → Consider feature engineering and model complexity increases")

        if avg_gap > 0.1:
            print("   🔴 Overall high overfitting tendency detected")
            print("      → Consider increasing regularization across all models")
            print("      → Evaluate if more training data is needed")
        elif avg_gap < 0.02:
            print("   🟢 Good generalization across models")
            print("      → Consider ensemble methods for further improvements")

        print(f"\n✅ Analysis complete! Use visualizations for detailed insights.")

    def export_analysis_results(self, filename="overfitting_analysis.csv", results_dir="output/results"):
        """Export analysis results to CSV in specified results directory with improved error handling"""

        if not self.analysis_results:
            print("❌ No analysis data available")
            return None

        # Create results directory with logging
        if not self._ensure_directory_exists(results_dir):
            print(f"❌ Failed to create results directory: {results_dir}")
            return None

        full_path = os.path.join(results_dir, filename)

        print(f"💾 Exporting overfitting analysis to {os.path.abspath(full_path)}...")

        try:
            export_data = []

            for model_key, analysis in self.analysis_results.items():
                model_name = analysis['model_name']
                train_perf = analysis['train_performance']
                test_perf = analysis['test_performance']
                diagnosis = analysis['fitting_diagnosis']
                complexity = analysis['complexity_analysis']

                row = {
                    'model': model_name,
                    'model_key': model_key,
                    'train_accuracy': train_perf['accuracy'],
                    'test_accuracy': test_perf['accuracy'],
                    'performance_gap': train_perf['accuracy'] - test_perf['accuracy'],
                    'fitting_status': diagnosis['fitting_status'],
                    'fitting_confidence': diagnosis['confidence'],
                    'samples_to_features_ratio': complexity.get('samples_to_features_ratio', 0),
                    'regularization': complexity.get('regularization', 0)
                }

                # Add model-specific complexity metrics
                if 'n_support_vectors' in complexity:
                    row['n_support_vectors'] = complexity['n_support_vectors']
                    row['support_vector_ratio'] = complexity['support_vector_ratio']
                if 'weight_magnitude' in complexity:
                    row['weight_magnitude'] = complexity['weight_magnitude']

                # Add training curve info
                training_curves = analysis['training_curves']
                if training_curves['available']:
                    row['training_converged'] = training_curves.get('converged', False)
                    row['training_epochs'] = training_curves.get('epochs_trained', 0)
                    row['training_improvement'] = training_curves.get('improvement_ratio', 0)

                # Add learning curve info
                learning_curves = analysis['learning_curves']
                if learning_curves['available']:
                    row['learning_curve_gap'] = learning_curves.get('gap_at_end', 0)

                export_data.append(row)

            # Create DataFrame and save
            df = pd.DataFrame(export_data)
            df.to_csv(full_path, index=False)

            # Verify file was created
            if os.path.exists(full_path):
                file_size = os.path.getsize(full_path)
                print(f"✅ Analysis results exported successfully: {os.path.abspath(full_path)} ({file_size} bytes)")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Rows: {len(df)}")
                return df
            else:
                print(f"❌ CSV file was not created: {full_path}")
                return None

        except Exception as e:
            print(f"❌ Error exporting analysis results: {e}")
            return None

    def create_comprehensive_analysis(self, figsize_1=(20, 15), figsize_2=(16, 12),
                                      save_plots=False, plots_dir='output/evaluation_plots', results_dir='output/results'):
        """Create all analysis visualizations and reports with explicit directory control and improved logging"""

        print("\n🎨 CREATING COMPREHENSIVE OVERFITTING/UNDERFITTING ANALYSIS")
        print("=" * 75)

        if not self.analysis_results:
            print("❌ No analysis data available. Run analyze_all_models() first.")
            return

        if save_plots:
            # Ensure both directories exist
            plots_success = self._ensure_directory_exists(plots_dir)
            results_success = self._ensure_directory_exists(results_dir)

            if not plots_success or not results_success:
                print("❌ Failed to create required directories, disabling save_plots")
                save_plots = False
            else:
                print(f"📁 Plots will be saved to: {os.path.abspath(plots_dir)}/")
                print(f"📁 Results will be saved to: {os.path.abspath(results_dir)}/")

        # Generate visualizations - all go to plots_dir
        print("\n🎨 Generating comprehensive fitting analysis plots...")
        self.plot_fitting_analysis(figsize_1, save_plots, plots_dir)

        print("\n📈 Generating detailed learning curves...")
        self.plot_detailed_learning_curves(figsize_2, save_plots, plots_dir)

        # Generate detailed report
        print("\n📋 Generating comprehensive text report...")
        self.generate_comprehensive_report()

        # Export results to results_dir
        if save_plots:
            print("\n💾 Exporting analysis results to CSV...")
            self.export_analysis_results("overfitting_analysis.csv", results_dir)

        if save_plots:
            print(f"\n💾 All plots saved to: {os.path.abspath(plots_dir)}/")
            print(f"💾 All results saved to: {os.path.abspath(results_dir)}/")

        print(f"\n🎉 Comprehensive overfitting/underfitting analysis complete!")


# Update the integration function
def integrate_overfitting_analysis(models_dict, X_train, y_train, X_test, y_test, model_names=None,
                                   save_plots=False, plots_dir="output/evaluation_plots", results_dir="output/results"):
    """
    Integration function for existing project structure with improved error handling

    Args:
        models_dict (dict): Dictionary of trained models
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_names (dict): Optional display names for models
        save_plots (bool): Whether to save plots and results
        plots_dir (str): Directory for plots
        results_dir (str): Directory for CSV files

    Returns:
        OverfittingAnalyzer: Analyzer with complete analysis
    """
    print("🔗 INTEGRATING OVERFITTING ANALYSIS WITH EXISTING MODELS")
    print("=" * 65)

    # Default model names
    default_names = {
        'lr_custom': 'Logistic Regression (Custom)',
        'svm_custom': 'Linear SVM (Custom)',
        'klr_custom': 'Kernel Logistic Regression (Custom)',
        'ksvm_custom': 'Kernel SVM (Custom)'
    }

    if model_names is None:
        model_names = default_names

    # Create analyzer
    analyzer = OverfittingAnalyzer()

    # Run analysis
    analyzer.analyze_all_models(models_dict, X_train, y_train, X_test, y_test, model_names)

    # Create comprehensive analysis with explicit directory control
    analyzer.create_comprehensive_analysis(save_plots=save_plots, plots_dir=plots_dir, results_dir=results_dir)

    return analyzer