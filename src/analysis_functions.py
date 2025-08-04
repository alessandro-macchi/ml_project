from src.visualization import integrate_with_experiment_results
from src.misclassification_analysis import MisclassificationAnalyzer
from src.overfitting_analysis import integrate_overfitting_analysis
from src.save import save_results
from src.utils import get_model_names, get_wine_feature_names


def run_comprehensive_analysis(results, trained_models, X_train, y_train, X_test, y_test, experiment_name):
    """
    Run comprehensive analysis including visualizations, misclassification analysis,
    and overfitting detection

    Args:
        results: Model performance results
        trained_models: Dictionary of trained model objects
        X_train, y_train: Training data
        X_test, y_test: Test data
        experiment_name: Name of the experiment

    Returns:
        tuple: (misclassification_analyzer, overfitting_analyzer)
    """

    # Save results to results directory
    save_results(results, experiment_name)

    # Standard visualizations - save to evaluation_plots
    run_visualizations(results, X_test, y_test)

    # Misclassification analysis - ensure consistent directory usage
    misclassification_analyzer = run_misclassification_analysis(trained_models, X_test, y_test)

    # Overfitting analysis - ensure consistent directory usage
    overfitting_analyzer = run_overfitting_analysis(trained_models, X_train, y_train, X_test, y_test)

    return misclassification_analyzer, overfitting_analyzer


def run_visualizations(results, X_test, y_test):
    """Generate standard performance visualizations"""
    print(f"\n{'=' * 70}")
    print("🎨 GENERATING PERFORMANCE VISUALIZATIONS")
    print(f"{'=' * 70}")

    visualizer = integrate_with_experiment_results(results, X_test, y_test)
    return visualizer


def run_misclassification_analysis(trained_models, X_test, y_test):
    """Run enhanced misclassification analysis with consistent directory structure"""
    print(f"\n{'=' * 70}")
    print("🔍 ENHANCED MISCLASSIFICATION ANALYSIS")
    print(f"{'=' * 70}")

    wine_feature_names = get_wine_feature_names()
    model_names = get_model_names()

    # Create analyzer with explicit save directory
    analyzer = MisclassificationAnalyzer(save_dir="evaluation_plots")
    analyzer.analyze_all_models(trained_models, X_test, y_test, wine_feature_names, model_names)

    # Create comprehensive analysis with plots saved to evaluation_plots
    analyzer.create_comprehensive_analysis(save_plots=True, save_dir="evaluation_plots")

    # Export results to results directory
    analyzer.export_analysis_results("wine_misclassification_analysis.csv", results_dir="results")

    return analyzer


def run_overfitting_analysis(trained_models, X_train, y_train, X_test, y_test):
    """Run overfitting/underfitting analysis with consistent directory structure"""
    print(f"\n{'=' * 70}")
    print("🎯 OVERFITTING/UNDERFITTING ANALYSIS")
    print(f"{'=' * 70}")

    model_names = get_model_names()
    overfitting_analyzer = integrate_overfitting_analysis(
        trained_models, X_train, y_train, X_test, y_test, model_names,
        save_plots=True, plots_dir="evaluation_plots", results_dir="results"
    )

    return overfitting_analyzer


def generate_final_summary_report(results, overfitting_analyzer, data):
    """Generate comprehensive final summary report"""
    print(f"\n{'=' * 80}")
    print("📋 FINAL SUMMARY REPORT")
    print(f"{'=' * 80}")

    model_names = get_model_names()

    # Best performing model
    generate_best_model_summary(results)

    # Overfitting summary
    generate_overfitting_summary(overfitting_analyzer, model_names)

    # Performance summary
    generate_performance_summary(results, model_names)

    # General recommendations
    generate_recommendations(overfitting_analyzer, model_names)

    # Data insights
    generate_data_insights(data)

    print(f"\n🎯 For detailed analysis, refer to the generated visualizations and CSV reports.")
    print(f"✅ All analyses complete!")


def generate_best_model_summary(results):
    """Generate summary of best performing model"""
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"🏆 Best performing model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")


def generate_overfitting_summary(overfitting_analyzer, model_names):
    """Generate overfitting status summary"""
    overfitting_summary = {}
    for model_key, analysis in overfitting_analyzer.analysis_results.items():
        status = analysis['fitting_diagnosis']['fitting_status']
        overfitting_summary[model_key] = status

    print(f"🎯 Overfitting status summary:")
    for model_key, status in overfitting_summary.items():
        emoji = {
            'overfitting': '🔴',
            'underfitting': '🟡',
            'good_fit': '🟢',
            'inconclusive': '⚪'
        }.get(status, '⚪')
        model_name = model_names.get(model_key, model_key)
        print(f"   {emoji} {model_name}: {status.upper()}")

    return overfitting_summary


def generate_performance_summary(results, model_names):
    """Generate performance summary for all models"""
    print(f"\n📈 Performance summary:")
    for model_key, metrics in results.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            model_name = model_names.get(model_key, model_key)
            accuracy = metrics['accuracy']
            f1 = metrics['f1']
            print(f"   📊 {model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")


def generate_recommendations(overfitting_analyzer, model_names):
    """Generate recommendations based on analysis results"""
    print(f"\n💡 FINAL RECOMMENDATIONS:")

    overfitting_summary = {}
    for model_key, analysis in overfitting_analyzer.analysis_results.items():
        status = analysis['fitting_diagnosis']['fitting_status']
        overfitting_summary[model_key] = status

    overfitting_models = [k for k, v in overfitting_summary.items() if v == 'overfitting']
    underfitting_models = [k for k, v in overfitting_summary.items() if v == 'underfitting']
    good_fit_models = [k for k, v in overfitting_summary.items() if v == 'good_fit']

    if good_fit_models:
        print(f"✅ Well-fitted models: {', '.join([model_names.get(k, k) for k in good_fit_models])}")
        print("   → These models are ready for production use")

    if overfitting_models:
        print(f"🔴 Overfitting detected in: {', '.join([model_names.get(k, k) for k in overfitting_models])}")
        print("   → Consider: increased regularization, more data, or simpler models")

    if underfitting_models:
        print(f"🟡 Underfitting detected in: {', '.join([model_names.get(k, k) for k in underfitting_models])}")
        print("   → Consider: reduced regularization, more complex models, or feature engineering")


def generate_data_insights(data):
    """Generate insights about the dataset"""
    print(f"\n📊 Dataset insights:")
    print(f"   • Total samples: {len(data)}")
    print(f"   • Features: {len(data.columns) - 1}")  # -1 for target variable
    print(f"   • Target distribution: {data['quality_binary'].value_counts().to_dict()}")
