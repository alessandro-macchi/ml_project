"""
Enhanced Misclassification Analysis Module with Plot Saving

This module provides deep analysis of misclassified examples to understand model limitations.
It goes beyond basic metrics to examine the characteristics of misclassified samples.
All plots are automatically saved to 'evaluation_plots' directory.

Usage:
    from src.misclassification_analysis import MisclassificationAnalyzer

    analyzer = MisclassificationAnalyzer()
    analyzer.analyze_all_models(models_dict, X_test, y_test, feature_names)
    analyzer.create_comprehensive_analysis()  # Saves plots automatically
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")


class MisclassificationAnalyzer:
    """
    Comprehensive misclassification analysis for understanding model limitations
    """

    def __init__(self, save_dir="evaluation_plots"):
        """
        Initialize the misclassification analyzer

        Args:
            save_dir (str): Directory to save analysis plots and results
        """
        self.save_dir = save_dir
        self.models = {}
        self.misclassified_data = {}
        self.feature_names = None
        self.X_test = None
        self.y_test = None

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def add_model_analysis(self, model_key, model, model_name=None):
        """
        Add a model for misclassification analysis

        Args:
            model_key (str): Unique identifier for the model
            model (object): Trained model object
            model_name (str): Display name for the model
        """
        self.models[model_key] = {
            'model': model,
            'name': model_name or model_key.replace('_', ' ').title()
        }

    def analyze_all_models(self, models_dict, X_test, y_test, feature_names=None, models_names=None):
        """
        Analyze misclassifications for all models

        Args:
            models_dict (dict): Dictionary of trained models
            X_test (array): Test features
            y_test (array): Test labels
            feature_names (list): Names of features
            models_names (dict): Display names for models
        """
        print("\n🔍 COMPREHENSIVE MISCLASSIFICATION ANALYSIS")
        print("=" * 60)

        # Store test data
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

        # Generate feature names if not provided
        if feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(self.X_test.shape[1])]
        else:
            self.feature_names = feature_names

        # Add models
        for model_key, model in models_dict.items():
            model_name = models_names.get(model_key) if models_names else None
            self.add_model_analysis(model_key, model, model_name)

        # Analyze each model
        for model_key, model_info in self.models.items():
            print(f"\n📊 Analyzing {model_info['name']}...")
            self._analyze_single_model(model_key)

        print("\n✅ Misclassification analysis complete!")

    def _analyze_single_model(self, model_key):
        """Analyze misclassifications for a single model"""
        model = self.models[model_key]['model']

        # Get predictions
        y_pred = model.predict(self.X_test)

        # Get prediction probabilities if available
        try:
            y_proba = model.predict_proba(self.X_test)
        except:
            try:
                y_scores = model.decision_function(self.X_test)
                y_proba = 1 / (1 + np.exp(-np.array(y_scores)))
            except:
                y_proba = None

        # Identify misclassified examples
        misclassified_mask = self.y_test != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]

        # Separate false positives and false negatives
        false_positives = np.where((self.y_test == 0) & (y_pred == 1))[0]
        false_negatives = np.where((self.y_test == 1) & (y_pred == 0))[0]

        # Analyze characteristics of misclassified examples
        analysis = self._analyze_misclassified_characteristics(
            misclassified_indices, false_positives, false_negatives, y_proba
        )

        # Store results
        self.misclassified_data[model_key] = {
            'predictions': y_pred,
            'probabilities': y_proba,
            'misclassified_indices': misclassified_indices,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'analysis': analysis
        }

        # Print summary
        self._print_model_summary(model_key, analysis)

    def _analyze_misclassified_characteristics(self, misclassified_indices, fp_indices, fn_indices, y_proba):
        """Analyze the characteristics of misclassified examples"""

        if len(misclassified_indices) == 0:
            return {
                'total_misclassified': 0,
                'feature_analysis': {},
                'confidence_analysis': {},
                'class_boundary_analysis': {}
            }

        # Basic statistics
        total_misclassified = len(misclassified_indices)
        total_samples = len(self.y_test)

        # 1. Feature Analysis: Compare feature distributions
        feature_analysis = self._analyze_feature_patterns(misclassified_indices, fp_indices, fn_indices)

        # 2. Confidence Analysis: Examine prediction confidence
        confidence_analysis = self._analyze_prediction_confidence(misclassified_indices, y_proba)

        # 3. Class Boundary Analysis: Examine samples near decision boundary
        boundary_analysis = self._analyze_class_boundaries(y_proba)

        return {
            'total_misclassified': total_misclassified,
            'misclassification_rate': total_misclassified / total_samples,
            'false_positives': len(fp_indices),
            'false_negatives': len(fn_indices),
            'fp_rate': len(fp_indices) / np.sum(self.y_test == 0) if np.sum(self.y_test == 0) > 0 else 0,
            'fn_rate': len(fn_indices) / np.sum(self.y_test == 1) if np.sum(self.y_test == 1) > 0 else 0,
            'feature_analysis': feature_analysis,
            'confidence_analysis': confidence_analysis,
            'class_boundary_analysis': boundary_analysis
        }

    def _analyze_feature_patterns(self, misclassified_indices, fp_indices, fn_indices):
        """Analyze feature patterns in misclassified examples"""

        if len(misclassified_indices) == 0:
            return {}

        correctly_classified = np.setdiff1d(np.arange(len(self.y_test)), misclassified_indices)

        feature_analysis = {}

        # For each feature, compare distributions
        for i, feature_name in enumerate(self.feature_names):
            feature_values = self.X_test[:, i]

            # Compare misclassified vs correctly classified
            misc_values = feature_values[misclassified_indices]
            correct_values = feature_values[correctly_classified]

            # Statistical tests
            try:
                # T-test for difference in means
                t_stat, t_pvalue = stats.ttest_ind(misc_values, correct_values)

                # Mann-Whitney U test (non-parametric)
                u_stat, u_pvalue = stats.mannwhitneyu(misc_values, correct_values, alternative='two-sided')

                feature_analysis[feature_name] = {
                    'misclassified_mean': np.mean(misc_values),
                    'misclassified_std': np.std(misc_values),
                    'correct_mean': np.mean(correct_values),
                    'correct_std': np.std(correct_values),
                    'mean_difference': np.mean(misc_values) - np.mean(correct_values),
                    't_test_pvalue': t_pvalue,
                    'mannwhitney_pvalue': u_pvalue,
                    'significant_difference': min(t_pvalue, u_pvalue) < 0.05
                }

                # Analyze false positives vs false negatives if we have both
                if len(fp_indices) > 0 and len(fn_indices) > 0:
                    fp_values = feature_values[fp_indices]
                    fn_values = feature_values[fn_indices]

                    fp_fn_t, fp_fn_p = stats.ttest_ind(fp_values, fn_values)

                    feature_analysis[feature_name].update({
                        'fp_mean': np.mean(fp_values),
                        'fn_mean': np.mean(fn_values),
                        'fp_fn_difference': np.mean(fp_values) - np.mean(fn_values),
                        'fp_fn_pvalue': fp_fn_p,
                        'fp_fn_significant': fp_fn_p < 0.05
                    })

            except Exception as e:
                feature_analysis[feature_name] = {
                    'error': str(e),
                    'misclassified_mean': np.mean(misc_values),
                    'correct_mean': np.mean(correct_values)
                }

        return feature_analysis

    def _analyze_prediction_confidence(self, misclassified_indices, y_proba):
        """Analyze prediction confidence for misclassified examples"""

        if y_proba is None or len(misclassified_indices) == 0:
            return {'available': False}

        correctly_classified = np.setdiff1d(np.arange(len(self.y_test)), misclassified_indices)

        # Calculate confidence (distance from 0.5)
        confidence_misc = np.abs(y_proba[misclassified_indices] - 0.5)
        confidence_correct = np.abs(y_proba[correctly_classified] - 0.5)

        return {
            'available': True,
            'misclassified_avg_confidence': np.mean(confidence_misc),
            'correct_avg_confidence': np.mean(confidence_correct),
            'confidence_difference': np.mean(confidence_correct) - np.mean(confidence_misc),
            'low_confidence_threshold': 0.1,  # Close to 0.5
            'low_confidence_misclassified': np.sum(confidence_misc < 0.1),
            'low_confidence_correct': np.sum(confidence_correct < 0.1),
            'high_confidence_misclassified': np.sum(confidence_misc > 0.4),
            'misclassified_proba_mean': np.mean(y_proba[misclassified_indices]),
            'misclassified_proba_std': np.std(y_proba[misclassified_indices])
        }

    def _analyze_class_boundaries(self, y_proba):
        """Analyze samples near class boundaries"""

        if y_proba is None:
            return {'available': False}

        # Define boundary region (e.g., probability between 0.4 and 0.6)
        boundary_mask = (y_proba >= 0.4) & (y_proba <= 0.6)
        boundary_indices = np.where(boundary_mask)[0]

        if len(boundary_indices) == 0:
            return {'available': True, 'boundary_samples': 0}

        # Analyze accuracy in boundary region
        boundary_predictions = (y_proba[boundary_indices] >= 0.5).astype(int)
        boundary_accuracy = np.mean(boundary_predictions == self.y_test[boundary_indices])

        return {
            'available': True,
            'boundary_samples': len(boundary_indices),
            'boundary_percentage': len(boundary_indices) / len(y_proba) * 100,
            'boundary_accuracy': boundary_accuracy,
            'boundary_error_rate': 1 - boundary_accuracy
        }

    def _print_model_summary(self, model_key, analysis):
        """Print summary of misclassification analysis for a model"""
        model_name = self.models[model_key]['name']

        print(f"   📈 {model_name} Summary:")
        print(
            f"      Total misclassified: {analysis['total_misclassified']} ({analysis['misclassification_rate']:.1%})")
        print(f"      False positives: {analysis['false_positives']} ({analysis['fp_rate']:.1%})")
        print(f"      False negatives: {analysis['false_negatives']} ({analysis['fn_rate']:.1%})")

        # Feature insights
        feature_analysis = analysis['feature_analysis']
        if feature_analysis:
            significant_features = [
                name for name, data in feature_analysis.items()
                if isinstance(data, dict) and data.get('significant_difference', False)
            ]

            if significant_features:
                print(f"      Features with significant differences: {len(significant_features)}")
                # Show top 3 most different features
                sorted_features = sorted(
                    [(name, abs(data.get('mean_difference', 0)))
                     for name, data in feature_analysis.items()
                     if isinstance(data, dict) and 'mean_difference' in data],
                    key=lambda x: x[1], reverse=True
                )[:3]

                for feature_name, diff in sorted_features:
                    print(f"         - {feature_name}: {diff:.3f} difference")

        # Confidence insights
        conf_analysis = analysis['confidence_analysis']
        if conf_analysis.get('available', False):
            print(f"      Avg confidence - Correct: {conf_analysis['correct_avg_confidence']:.3f}, "
                  f"Misclassified: {conf_analysis['misclassified_avg_confidence']:.3f}")

            if conf_analysis['low_confidence_misclassified'] > 0:
                print(f"      Low-confidence misclassifications: {conf_analysis['low_confidence_misclassified']}")

    def plot_feature_importance_misclassification(self, figsize=(16, 10), save_plots=False, save_dir=None):
        """
        Plot feature importance in misclassification

        Args:
            figsize (tuple): Figure size for the plot
            save_plots (bool): Whether to save the plot
            save_dir (str): Directory to save plots (overrides default)
        """
        print("📊 Creating Feature Misclassification Analysis...")

        if not self.misclassified_data:
            print("❌ No misclassification data available")
            return

        # Find features with significant differences across all models
        all_feature_impacts = {}

        for model_key, data in self.misclassified_data.items():
            feature_analysis = data['analysis']['feature_analysis']

            for feature_name, analysis in feature_analysis.items():
                if isinstance(analysis, dict) and 'mean_difference' in analysis:
                    if feature_name not in all_feature_impacts:
                        all_feature_impacts[feature_name] = []
                    all_feature_impacts[feature_name].append({
                        'model': self.models[model_key]['name'],
                        'difference': abs(analysis['mean_difference']),
                        'significant': analysis.get('significant_difference', False),
                        'pvalue': analysis.get('t_test_pvalue', 1.0)
                    })

        if not all_feature_impacts:
            print("❌ No feature analysis data available")
            return

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Feature impact heatmap
        ax1 = axes[0, 0]

        # Prepare data for heatmap
        models = list(self.models.keys())
        model_names = [self.models[key]['name'].replace(' (Custom)', '') for key in models]

        # Get top features by average impact
        feature_avg_impact = {
            feature: np.mean([item['difference'] for item in impacts])
            for feature, impacts in all_feature_impacts.items()
        }

        top_features = sorted(feature_avg_impact.items(), key=lambda x: x[1], reverse=True)[:10]
        top_feature_names = [item[0] for item in top_features]

        # Create heatmap data
        heatmap_data = []
        for model_key in models:
            model_row = []
            feature_analysis = self.misclassified_data[model_key]['analysis']['feature_analysis']

            for feature_name in top_feature_names:
                if feature_name in feature_analysis and isinstance(feature_analysis[feature_name], dict):
                    diff = abs(feature_analysis[feature_name].get('mean_difference', 0))
                    model_row.append(diff)
                else:
                    model_row.append(0)
            heatmap_data.append(model_row)

        heatmap_data = np.array(heatmap_data)

        im = ax1.imshow(heatmap_data, cmap='Reds', aspect='auto')
        ax1.set_xticks(range(len(top_feature_names)))
        ax1.set_yticks(range(len(model_names)))
        ax1.set_xticklabels(top_feature_names, rotation=45, ha='right')
        ax1.set_yticklabels(model_names)
        ax1.set_title('Feature Impact on Misclassification\n(Absolute Mean Difference)', fontweight='bold')

        # Add text annotations
        for i in range(len(model_names)):
            for j in range(len(top_feature_names)):
                text = ax1.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                ha="center", va="center", color="black" if heatmap_data[i, j] < 0.5 else "white")

        plt.colorbar(im, ax=ax1, shrink=0.8)

        # 2. Significance count
        ax2 = axes[0, 1]

        # Count significant features per model
        sig_counts = []
        for model_key in models:
            feature_analysis = self.misclassified_data[model_key]['analysis']['feature_analysis']
            sig_count = sum(1 for analysis in feature_analysis.values()
                            if isinstance(analysis, dict) and analysis.get('significant_difference', False))
            sig_counts.append(sig_count)

        bars = ax2.bar(range(len(model_names)), sig_counts, color='lightblue', alpha=0.8)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Number of Significant Features')
        ax2.set_title('Significant Features in Misclassification', fontweight='bold')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom')

        # 3. Confidence analysis
        ax3 = axes[1, 0]

        conf_data = []
        model_names_conf = []

        for model_key in models:
            conf_analysis = self.misclassified_data[model_key]['analysis']['confidence_analysis']
            if conf_analysis.get('available', False):
                conf_data.append([
                    conf_analysis['correct_avg_confidence'],
                    conf_analysis['misclassified_avg_confidence']
                ])
                model_names_conf.append(self.models[model_key]['name'].replace(' (Custom)', ''))

        if conf_data:
            conf_data = np.array(conf_data)
            x = np.arange(len(model_names_conf))
            width = 0.35

            bars1 = ax3.bar(x - width / 2, conf_data[:, 0], width, label='Correct', alpha=0.8, color='lightgreen')
            bars2 = ax3.bar(x + width / 2, conf_data[:, 1], width, label='Misclassified', alpha=0.8, color='lightcoral')

            ax3.set_xlabel('Models')
            ax3.set_ylabel('Average Confidence')
            ax3.set_title('Prediction Confidence Analysis', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(model_names_conf, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No confidence data available',
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Prediction Confidence Analysis', fontweight='bold')

        # 4. Error type distribution
        ax4 = axes[1, 1]

        fp_counts = []
        fn_counts = []

        for model_key in models:
            analysis = self.misclassified_data[model_key]['analysis']
            fp_counts.append(analysis['false_positives'])
            fn_counts.append(analysis['false_negatives'])

        x = np.arange(len(model_names))
        width = 0.35

        bars3 = ax4.bar(x - width / 2, fp_counts, width, label='False Positives', alpha=0.8, color='orange')
        bars4 = ax4.bar(x + width / 2, fn_counts, width, label='False Negatives', alpha=0.8, color='red')

        ax4.set_xlabel('Models')
        ax4.set_ylabel('Count')
        ax4.set_title('Error Type Distribution', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{int(height)}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.suptitle('Comprehensive Misclassification Analysis', fontsize=16, fontweight='bold', y=1.02)

        # Save plot if requested
        if save_plots:
            save_directory = save_dir or self.save_dir
            os.makedirs(save_directory, exist_ok=True)
            filename = os.path.join(save_directory, 'misclassification_feature_importance.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved: {filename}")

        plt.show()

    def plot_misclassified_examples_distribution(self, figsize=(16, 12), save_plots=False, save_dir=None):
        """
        Plot distribution of misclassified examples in feature space

        Args:
            figsize (tuple): Figure size for the plot
            save_plots (bool): Whether to save the plot
            save_dir (str): Directory to save plots (overrides default)
        """
        print("📊 Creating Misclassified Examples Distribution...")

        if not self.misclassified_data or self.X_test is None:
            print("❌ No data available for plotting")
            return

        # Select top 4 most important features for visualization
        feature_importance = {}

        for model_key, data in self.misclassified_data.items():
            feature_analysis = data['analysis']['feature_analysis']

            for feature_name, analysis in feature_analysis.items():
                if isinstance(analysis, dict) and 'mean_difference' in analysis:
                    if feature_name not in feature_importance:
                        feature_importance[feature_name] = []
                    feature_importance[feature_name].append(abs(analysis['mean_difference']))

        # Get top 4 features by average importance
        top_features = sorted(
            [(name, np.mean(impacts)) for name, impacts in feature_importance.items()],
            key=lambda x: x[1], reverse=True
        )[:4]

        if len(top_features) < 2:
            print("❌ Not enough features for distribution analysis")
            return

        # Create subplot grid
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        # Take first model for demonstration
        first_model_key = list(self.misclassified_data.keys())[0]
        misclassified_indices = self.misclassified_data[first_model_key]['misclassified_indices']
        fp_indices = self.misclassified_data[first_model_key]['false_positives']
        fn_indices = self.misclassified_data[first_model_key]['false_negatives']

        correctly_classified = np.setdiff1d(np.arange(len(self.y_test)), misclassified_indices)

        # Plot distributions for top features
        for i, (feature_name, importance) in enumerate(top_features):
            if i >= 4:
                break

            ax = axes[i]

            # Get feature index
            try:
                feature_idx = self.feature_names.index(feature_name)
            except ValueError:
                continue

            feature_values = self.X_test[:, feature_idx]

            # Create distributions
            correct_values = feature_values[correctly_classified]
            fp_values = feature_values[fp_indices] if len(fp_indices) > 0 else []
            fn_values = feature_values[fn_indices] if len(fn_indices) > 0 else []

            # Plot histograms
            ax.hist(correct_values, alpha=0.6, bins=20, label='Correct', color='lightgreen', density=True)

            if len(fp_values) > 0:
                ax.hist(fp_values, alpha=0.8, bins=15, label='False Positives', color='orange', density=True)

            if len(fn_values) > 0:
                ax.hist(fn_values, alpha=0.8, bins=15, label='False Negatives', color='red', density=True)

            # Add vertical lines for means
            ax.axvline(np.mean(correct_values), color='green', linestyle='--', alpha=0.8,
                       label=f'Correct Mean: {np.mean(correct_values):.3f}')

            if len(fp_values) > 0:
                ax.axvline(np.mean(fp_values), color='orange', linestyle='--', alpha=0.8,
                           label=f'FP Mean: {np.mean(fp_values):.3f}')

            if len(fn_values) > 0:
                ax.axvline(np.mean(fn_values), color='red', linestyle='--', alpha=0.8,
                           label=f'FN Mean: {np.mean(fn_values):.3f}')

            ax.set_xlabel(feature_name)
            ax.set_ylabel('Density')
            ax.set_title(f'{feature_name} Distribution\n(Importance: {importance:.3f})', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(f'Feature Distributions: Correct vs Misclassified\n({self.models[first_model_key]["name"]})',
                     fontsize=16, fontweight='bold', y=1.02)

        # Save plot if requested
        if save_plots:
            save_directory = save_dir or self.save_dir
            os.makedirs(save_directory, exist_ok=True)
            filename = os.path.join(save_directory, 'misclassified_examples_distribution.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved: {filename}")

        plt.show()

    def plot_confidence_analysis(self, figsize=(14, 10), save_plots=False, save_dir=None):
        """
        Create detailed confidence analysis plots

        Args:
            figsize (tuple): Figure size for the plot
            save_plots (bool): Whether to save the plot
            save_dir (str): Directory to save plots (overrides default)
        """
        print("📊 Creating Confidence Analysis...")

        if not self.misclassified_data:
            print("❌ No misclassification data available")
            return

        # Filter models with confidence data
        models_with_conf = {}
        for model_key, data in self.misclassified_data.items():
            if data['analysis']['confidence_analysis'].get('available', False):
                models_with_conf[model_key] = data

        if not models_with_conf:
            print("❌ No confidence data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Confidence distributions
        ax1 = axes[0, 0]
        first_model = list(models_with_conf.keys())[0]
        probabilities = models_with_conf[first_model]['probabilities']
        misclassified_indices = models_with_conf[first_model]['misclassified_indices']

        correctly_classified = np.setdiff1d(np.arange(len(self.y_test)), misclassified_indices)

        # Plot probability distributions
        ax1.hist(probabilities[correctly_classified], alpha=0.6, bins=30, label='Correct',
                color='lightgreen', density=True)
        ax1.hist(probabilities[misclassified_indices], alpha=0.8, bins=20, label='Misclassified',
                color='red', density=True)

        # Add decision boundary
        ax1.axvline(0.5, color='black', linestyle='--', alpha=0.7, label='Decision Boundary')

        ax1.set_xlabel('Prediction Probability')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Prediction Probability Distribution\n({self.models[first_model]["name"]})',
                     fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Confidence vs Accuracy scatter
        ax2 = axes[0, 1]

        # Create confidence bins and calculate accuracy for each
        conf_bins = np.linspace(0, 0.5, 11)  # Confidence is distance from 0.5
        bin_accuracies = []
        bin_centers = []
        bin_counts = []

        for i in range(len(conf_bins) - 1):
            conf_lower = conf_bins[i]
            conf_upper = conf_bins[i + 1]

            # Find samples in this confidence range
            confidence_all = np.abs(probabilities - 0.5)
            in_bin = (confidence_all >= conf_lower) & (confidence_all < conf_upper)

            if np.sum(in_bin) > 0:
                predictions = (probabilities[in_bin] >= 0.5).astype(int)
                accuracy = np.mean(predictions == self.y_test[in_bin])

                bin_accuracies.append(accuracy)
                bin_centers.append((conf_lower + conf_upper) / 2)
                bin_counts.append(np.sum(in_bin))

        if bin_centers:
            # Size points by number of samples
            sizes = [max(20, min(200, count * 5)) for count in bin_counts]
            scatter = ax2.scatter(bin_centers, bin_accuracies, s=sizes, alpha=0.7, c=bin_counts,
                                cmap='viridis')

            # Add trend line
            if len(bin_centers) > 1:
                z = np.polyfit(bin_centers, bin_accuracies, 1)
                p = np.poly1d(z)
                ax2.plot(bin_centers, p(bin_centers), "r--", alpha=0.8, label='Trend')

            ax2.set_xlabel('Confidence (Distance from 0.5)')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Confidence vs Accuracy', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax2, shrink=0.8, label='Sample Count')

        # 3. Boundary analysis
        ax3 = axes[1, 0]

        boundary_data = []
        boundary_models = []

        for model_key in models_with_conf.keys():
            boundary_analysis = models_with_conf[model_key]['analysis']['class_boundary_analysis']
            if boundary_analysis.get('available', False) and boundary_analysis.get('boundary_samples', 0) > 0:
                boundary_data.append([
                    boundary_analysis['boundary_percentage'],
                    boundary_analysis['boundary_accuracy'] * 100
                ])
                boundary_models.append(self.models[model_key]['name'].replace(' (Custom)', ''))

        if boundary_data:
            boundary_data = np.array(boundary_data)
            x = np.arange(len(boundary_models))
            width = 0.35

            bars1 = ax3.bar(x - width/2, boundary_data[:, 0], width, label='% in Boundary', alpha=0.8, color='lightblue')
            bars2 = ax3.bar(x + width/2, boundary_data[:, 1], width, label='Boundary Accuracy %', alpha=0.8, color='orange')

            ax3.set_xlabel('Models')
            ax3.set_ylabel('Percentage')
            ax3.set_title('Decision Boundary Analysis', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(boundary_models, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width() / 2., height + 1,
                             f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        # 4. Confidence comparison across models
        ax4 = axes[1, 1]

        conf_comparison = []
        comp_models = []

        for model_key in models_with_conf.keys():
            conf_analysis = models_with_conf[model_key]['analysis']['confidence_analysis']
            conf_comparison.append([
                conf_analysis['correct_avg_confidence'],
                conf_analysis['misclassified_avg_confidence'],
                conf_analysis['confidence_difference']
            ])
            comp_models.append(self.models[model_key]['name'].replace(' (Custom)', ''))

        if conf_comparison:
            conf_comparison = np.array(conf_comparison)

            # Plot confidence difference
            colors = ['green' if diff > 0 else 'red' for diff in conf_comparison[:, 2]]
            bars = ax4.bar(range(len(comp_models)), conf_comparison[:, 2], color=colors, alpha=0.7)

            ax4.set_xlabel('Models')
            ax4.set_ylabel('Confidence Difference\n(Correct - Misclassified)')
            ax4.set_title('Model Confidence Discrimination', fontweight='bold')
            ax4.set_xticks(range(len(comp_models)))
            ax4.set_xticklabels(comp_models, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2.,
                         height + 0.005 if height >= 0 else height - 0.01,
                         f'{height:.3f}', ha='center',
                         va='bottom' if height >= 0 else 'top', fontsize=8)

        plt.tight_layout()
        plt.suptitle('Detailed Confidence Analysis', fontsize=16, fontweight='bold', y=1.02)

        # Save plot if requested
        if save_plots:
            save_directory = save_dir or self.save_dir
            os.makedirs(save_directory, exist_ok=True)
            filename = os.path.join(save_directory, 'misclassification_confidence_analysis.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved: {filename}")

        plt.show()

    def generate_detailed_report(self):
        """Generate comprehensive text report of misclassification analysis"""

        print("\n📋 COMPREHENSIVE MISCLASSIFICATION ANALYSIS REPORT")
        print("=" * 70)

        if not self.misclassified_data:
            print("❌ No analysis data available")
            return

        # Overall statistics
        print("\n🔍 OVERALL FINDINGS:")
        print("-" * 25)

        total_models = len(self.misclassified_data)
        total_samples = len(self.y_test)

        print(f"   Models analyzed: {total_models}")
        print(f"   Total test samples: {total_samples}")

        # Model-specific analysis
        print(f"\n📊 MODEL-SPECIFIC ANALYSIS:")
        print("-" * 30)

        for model_key, data in self.misclassified_data.items():
            model_name = self.models[model_key]['name']
            analysis = data['analysis']

            print(f"\n🔹 {model_name}:")
            print(f"   Total misclassified: {analysis['total_misclassified']} ({analysis['misclassification_rate']:.1%})")
            print(f"   False positives: {analysis['false_positives']} ({analysis['fp_rate']:.1%})")
            print(f"   False negatives: {analysis['false_negatives']} ({analysis['fn_rate']:.1%})")

            # Feature analysis summary
            feature_analysis = analysis['feature_analysis']
            if feature_analysis:
                significant_features = [
                    (name, data.get('mean_difference', 0))
                    for name, data in feature_analysis.items()
                    if isinstance(data, dict) and data.get('significant_difference', False)
                ]

                print(f"   Significant features: {len(significant_features)}")

                if significant_features:
                    # Show top 3 most impactful features
                    sorted_features = sorted(significant_features, key=lambda x: abs(x[1]), reverse=True)[:3]
                    for feature_name, diff in sorted_features:
                        print(f"      • {feature_name}: {diff:.3f} mean difference")

            # Confidence analysis summary
            conf_analysis = analysis['confidence_analysis']
            if conf_analysis.get('available', False):
                print(f"   Confidence analysis:")
                print(f"      • Correct avg confidence: {conf_analysis['correct_avg_confidence']:.3f}")
                print(f"      • Misclassified avg confidence: {conf_analysis['misclassified_avg_confidence']:.3f}")
                print(f"      • Confidence difference: {conf_analysis['confidence_difference']:.3f}")

                if conf_analysis['low_confidence_misclassified'] > 0:
                    print(f"      • Low-confidence errors: {conf_analysis['low_confidence_misclassified']}")

            # Boundary analysis summary
            boundary_analysis = analysis['class_boundary_analysis']
            if boundary_analysis.get('available', False) and boundary_analysis.get('boundary_samples', 0) > 0:
                print(f"   Decision boundary analysis:")
                print(f"      • Samples in boundary region: {boundary_analysis['boundary_samples']} ({boundary_analysis['boundary_percentage']:.1f}%)")
                print(f"      • Boundary region accuracy: {boundary_analysis['boundary_accuracy']:.3f}")

        # Cross-model insights
        print(f"\n🎯 CROSS-MODEL INSIGHTS:")
        print("-" * 25)

        # Find most problematic features across all models
        all_significant_features = {}
        for model_key, data in self.misclassified_data.items():
            feature_analysis = data['analysis']['feature_analysis']
            for feature_name, analysis in feature_analysis.items():
                if isinstance(analysis, dict) and analysis.get('significant_difference', False):
                    if feature_name not in all_significant_features:
                        all_significant_features[feature_name] = 0
                    all_significant_features[feature_name] += 1

        common_problematic = [
            (feature, count) for feature, count in all_significant_features.items()
            if count > 1
        ]

        if common_problematic:
            print(f"   Features problematic across multiple models:")
            for feature, count in sorted(common_problematic, key=lambda x: x[1], reverse=True):
                print(f"      • {feature}: problematic in {count}/{total_models} models")

        # Model ranking by error characteristics
        model_scores = []
        for model_key, data in self.misclassified_data.items():
            analysis = data['analysis']

            # Simple scoring: lower misclassification rate + balanced errors
            misc_rate = analysis['misclassification_rate']
            fp_rate = analysis['fp_rate']
            fn_rate = analysis['fn_rate']
            error_balance = abs(fp_rate - fn_rate)  # Lower is better (more balanced)

            # Confidence discrimination (if available)
            conf_analysis = analysis['confidence_analysis']
            conf_score = 0
            if conf_analysis.get('available', False):
                conf_score = conf_analysis.get('confidence_difference', 0)  # Higher is better

            # Overall score (lower is better for rates, higher for confidence)
            overall_score = -(misc_rate + error_balance) + conf_score

            model_scores.append((
                self.models[model_key]['name'],
                overall_score,
                misc_rate,
                error_balance,
                conf_score
            ))

        model_scores.sort(key=lambda x: x[1], reverse=True)

        print(f"\n   Model ranking by error characteristics:")
        for i, (model_name, score, misc_rate, balance, conf) in enumerate(model_scores, 1):
            print(f"      {i}. {model_name}")
            print(f"         • Misclassification rate: {misc_rate:.3f}")
            print(f"         • Error balance: {balance:.3f}")
            print(f"         • Confidence discrimination: {conf:.3f}")

        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 20)

        # Generate recommendations based on analysis
        if common_problematic:
            print(f"   📊 Feature Engineering:")
            print(f"      • Investigate features: {', '.join([f[0] for f in common_problematic[:3]])}")
            print(f"      • Consider feature transformation or selection")

        # Model-specific recommendations
        worst_model = model_scores[-1] if model_scores else None
        best_model = model_scores[0] if model_scores else None

        if worst_model and best_model:
            print(f"   🎯 Model Improvement:")
            print(f"      • Focus on improving: {worst_model[0]}")
            print(f"      • Study techniques from: {best_model[0]}")

        # Confidence-based recommendations
        low_conf_models = [
            self.models[model_key]['name']
            for model_key, data in self.misclassified_data.items()
            if data['analysis']['confidence_analysis'].get('available', False)
            and data['analysis']['confidence_analysis'].get('confidence_difference', 0) < 0.1
        ]

        if low_conf_models:
            print(f"   🔍 Confidence Improvement:")
            print(f"      • Models with poor confidence discrimination: {', '.join(low_conf_models)}")
            print(f"      • Consider calibration techniques")

        print(f"\n✅ Analysis complete!")

    def export_analysis_results(self, filename="misclassification_analysis.csv"):
        """Export misclassification analysis results to CSV"""

        if not self.misclassified_data:
            print("❌ No analysis data available")
            return None

        # Ensure results directory exists and use it for file path
        results_dir = os.path.join(self.save_dir, "../results")
        os.makedirs(results_dir, exist_ok=True)
        full_path = os.path.join(results_dir, filename)

        print(f"💾 Exporting misclassification analysis to {full_path}...")

        export_data = []

        for model_key, data in self.misclassified_data.items():
            model_name = self.models[model_key]['name']
            analysis = data['analysis']

            # Basic metrics
            row = {
                'model': model_name,
                'model_key': model_key,
                'total_misclassified': analysis['total_misclassified'],
                'misclassification_rate': analysis['misclassification_rate'],
                'false_positives': analysis['false_positives'],
                'false_negatives': analysis['false_negatives'],
                'fp_rate': analysis['fp_rate'],
                'fn_rate': analysis['fn_rate']
            }

            # Confidence metrics
            conf_analysis = analysis['confidence_analysis']
            if conf_analysis.get('available', False):
                row.update({
                    'correct_avg_confidence': conf_analysis['correct_avg_confidence'],
                    'misclassified_avg_confidence': conf_analysis['misclassified_avg_confidence'],
                    'confidence_difference': conf_analysis['confidence_difference'],
                    'low_confidence_misclassified': conf_analysis['low_confidence_misclassified']
                })

            # Boundary analysis
            boundary_analysis = analysis['class_boundary_analysis']
            if boundary_analysis.get('available', False):
                row.update({
                    'boundary_samples': boundary_analysis.get('boundary_samples', 0),
                    'boundary_percentage': boundary_analysis.get('boundary_percentage', 0),
                    'boundary_accuracy': boundary_analysis.get('boundary_accuracy', 0)
                })

            # Feature analysis summary
            feature_analysis = analysis['feature_analysis']
            if feature_analysis:
                significant_count = sum(1 for data in feature_analysis.values()
                                        if isinstance(data, dict) and data.get('significant_difference', False))
                row['significant_features_count'] = significant_count

                # Get most impactful feature
                max_impact_feature = max(
                    [(name, abs(data.get('mean_difference', 0)))
                     for name, data in feature_analysis.items()
                     if isinstance(data, dict) and 'mean_difference' in data],
                    key=lambda x: x[1],
                    default=('None', 0)
                )
                row['most_impactful_feature'] = max_impact_feature[0]
                row['max_feature_impact'] = max_impact_feature[1]

            export_data.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(full_path, index=False)

        print(f"✅ Analysis results exported to {full_path}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Rows: {len(df)}")

        return df

    def create_comprehensive_analysis(self, save_plots=False, save_dir=None):
        """
        Create all analysis visualizations and reports

        Args:
            save_plots (bool): Whether to save plots
            save_dir (str): Directory to save plots (overrides default)
        """
        print("\n🎨 CREATING COMPREHENSIVE MISCLASSIFICATION ANALYSIS")
        print("=" * 65)

        if not self.misclassified_data:
            print("❌ No analysis data available. Run analyze_all_models() first.")
            return

        # Use provided save_dir or default
        final_save_dir = save_dir or self.save_dir

        if save_plots:
            os.makedirs(final_save_dir, exist_ok=True)
            print(f"📁 Plots will be saved to: {final_save_dir}/")

        # Generate visualizations
        self.plot_feature_importance_misclassification(save_plots=save_plots, save_dir=final_save_dir)
        self.plot_misclassified_examples_distribution(save_plots=save_plots, save_dir=final_save_dir)
        self.plot_confidence_analysis(save_plots=save_plots, save_dir=final_save_dir)

        # Generate detailed report
        self.generate_detailed_report()

        if save_plots:
            print(f"\n💾 All plots saved to: {final_save_dir}/")

        print(f"\n🎉 Comprehensive misclassification analysis complete!")


# Integration functions
def integrate_misclassification_analysis(models_dict, X_test, y_test, feature_names=None, model_names=None, save_plots=False):
    """
    Integration function for existing project structure

    Args:
        models_dict (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        feature_names (list): Names of features
        model_names (dict): Optional display names for models
        save_plots (bool): Whether to save plots to evaluation_plots directory

    Returns:
        MisclassificationAnalyzer: Analyzer with complete analysis
    """
    print("🔗 INTEGRATING MISCLASSIFICATION ANALYSIS WITH EXISTING MODELS")
    print("=" * 70)

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
    analyzer = MisclassificationAnalyzer()

    # Run analysis
    analyzer.analyze_all_models(models_dict, X_test, y_test, feature_names, model_names)

    # Create comprehensive analysis with plot saving option
    analyzer.create_comprehensive_analysis(save_plots=save_plots)

    return analyzer