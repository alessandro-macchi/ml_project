import os
from datetime import datetime
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
    analyzer = MisclassificationAnalyzer(save_dir="output/evaluation_plots")
    analyzer.analyze_all_models(trained_models, X_test, y_test, wine_feature_names, model_names)

    # Create comprehensive analysis with plots saved to evaluation_plots
    analyzer.create_comprehensive_analysis(save_plots=True, save_dir="output/evaluation_plots")

    # Export results to results directory
    analyzer.export_analysis_results("wine_misclassification_analysis.csv", results_dir="output/results")

    return analyzer


def run_overfitting_analysis(trained_models, X_train, y_train, X_test, y_test):
    """Run overfitting/underfitting analysis with consistent directory structure"""
    print(f"\n{'=' * 70}")
    print("🎯 OVERFITTING/UNDERFITTING ANALYSIS")
    print(f"{'=' * 70}")

    model_names = get_model_names()
    overfitting_analyzer = integrate_overfitting_analysis(
        trained_models, X_train, y_train, X_test, y_test, model_names,
        save_plots=True, plots_dir="output/evaluation_plots", results_dir="output/results"
    )

    return overfitting_analyzer


def generate_final_summary_report(results, overfitting_analyzer, data):
    """Generate comprehensive final summary report and save to file"""
    print(f"\n{'=' * 80}")
    print("📋 FINAL SUMMARY REPORT")
    print(f"{'=' * 80}")

    model_names = get_model_names()

    # Create output directory if it doesn't exist
    output_dir = os.path.join("output", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"final_summary_report_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_filename)

    # Collect all report content
    report_content = []
    report_content.append("=" * 80)
    report_content.append("WINE QUALITY CLASSIFICATION - FINAL SUMMARY REPORT")
    report_content.append("=" * 80)
    report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("")

    # Best performing model
    best_model_content = generate_best_model_summary(results)
    report_content.extend(best_model_content)

    # Overfitting summary
    overfitting_content = generate_overfitting_summary(overfitting_analyzer, model_names)
    report_content.extend(overfitting_content)

    # Performance summary
    performance_content = generate_performance_summary(results, model_names)
    report_content.extend(performance_content)

    # General recommendations
    recommendations_content = generate_recommendations(overfitting_analyzer, model_names)
    report_content.extend(recommendations_content)

    # Data insights
    data_insights_content = generate_data_insights(data)
    report_content.extend(data_insights_content)

    report_content.append("")
    report_content.append("🎯 For detailed analysis, refer to the generated visualizations and CSV reports.")
    report_content.append("✅ All analyses complete!")

    # Save report to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))

    # Print to terminal (original behavior)
    for line in report_content:
        print(line)

    print(f"\n💾 Final summary report saved to: {report_path}")

    return report_path


def generate_best_model_summary(results):
    """Generate summary of best performing model"""
    content = []
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    content.append("🏆 BEST PERFORMING MODEL")
    content.append("-" * 30)
    content.append(f"Model: {best_model[0]}")
    content.append(f"Accuracy: {best_model[1]['accuracy']:.4f}")
    content.append(f"F1 Score: {best_model[1]['f1']:.4f}")
    content.append(f"Precision: {best_model[1]['precision']:.4f}")
    content.append(f"Recall: {best_model[1]['recall']:.4f}")
    content.append("")

    # Also print to terminal
    print(f"🏆 Best performing model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

    return content


def generate_overfitting_summary(overfitting_analyzer, model_names):
    """Generate overfitting status summary"""
    content = []
    overfitting_summary = {}

    for model_key, analysis in overfitting_analyzer.analysis_results.items():
        status = analysis['fitting_diagnosis']['fitting_status']
        overfitting_summary[model_key] = status

    content.append("🎯 OVERFITTING STATUS SUMMARY")
    content.append("-" * 35)

    print(f"🎯 Overfitting status summary:")

    for model_key, status in overfitting_summary.items():
        emoji = {
            'overfitting': '🔴',
            'underfitting': '🟡',
            'good_fit': '🟢',
            'inconclusive': '⚪'
        }.get(status, '⚪')
        model_name = model_names.get(model_key, model_key)

        # Add to file content
        content.append(f"{emoji} {model_name}: {status.upper()}")

        # Print to terminal
        print(f"   {emoji} {model_name}: {status.upper()}")

    content.append("")
    return content


def generate_performance_summary(results, model_names):
    """Generate performance summary for all models"""
    content = []
    content.append("📈 PERFORMANCE SUMMARY")
    content.append("-" * 25)

    print(f"\n📈 Performance summary:")

    for model_key, metrics in results.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            model_name = model_names.get(model_key, model_key)
            accuracy = metrics['accuracy']
            f1 = metrics['f1']
            precision = metrics['precision']
            recall = metrics['recall']

            # Add to file content
            content.append(f"📊 {model_name}:")
            content.append(f"   - Accuracy: {accuracy:.4f}")
            content.append(f"   - F1 Score: {f1:.4f}")
            content.append(f"   - Precision: {precision:.4f}")
            content.append(f"   - Recall: {recall:.4f}")

            # Print to terminal
            print(f"   📊 {model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

    content.append("")
    return content


def generate_recommendations(overfitting_analyzer, model_names):
    """Generate recommendations based on analysis results"""
    content = []
    content.append("💡 FINAL RECOMMENDATIONS")
    content.append("-" * 25)

    overfitting_summary = {}
    for model_key, analysis in overfitting_analyzer.analysis_results.items():
        status = analysis['fitting_diagnosis']['fitting_status']
        overfitting_summary[model_key] = status

    overfitting_models = [k for k, v in overfitting_summary.items() if v == 'overfitting']
    underfitting_models = [k for k, v in overfitting_summary.items() if v == 'underfitting']
    good_fit_models = [k for k, v in overfitting_summary.items() if v == 'good_fit']

    print(f"\n💡 FINAL RECOMMENDATIONS:")

    if good_fit_models:
        good_fit_names = [model_names.get(k, k) for k in good_fit_models]
        content.append(f"✅ Well-fitted models: {', '.join(good_fit_names)}")
        content.append("   → These models are ready for production use")

        print(f"✅ Well-fitted models: {', '.join(good_fit_names)}")
        print("   → These models are ready for production use")

    if overfitting_models:
        overfitting_names = [model_names.get(k, k) for k in overfitting_models]
        content.append(f"🔴 Overfitting detected in: {', '.join(overfitting_names)}")
        content.append("   → Consider: increased regularization, more data, or simpler models")

        print(f"🔴 Overfitting detected in: {', '.join(overfitting_names)}")
        print("   → Consider: increased regularization, more data, or simpler models")

    if underfitting_models:
        underfitting_names = [model_names.get(k, k) for k in underfitting_models]
        content.append(f"🟡 Underfitting detected in: {', '.join(underfitting_names)}")
        content.append("   → Consider: reduced regularization, more complex models, or feature engineering")

        print(f"🟡 Underfitting detected in: {', '.join(underfitting_names)}")
        print("   → Consider: reduced regularization, more complex models, or feature engineering")

    content.append("")
    return content


def generate_data_insights(data):
    """Generate insights about the dataset"""
    content = []
    content.append("📊 DATASET INSIGHTS")
    content.append("-" * 20)
    content.append(f"• Total samples: {len(data)}")
    content.append(f"• Features: {len(data.columns) - 1}")  # -1 for target variable

    target_dist = data['quality_binary'].value_counts().to_dict()
    content.append(f"• Target distribution: {target_dist}")

    # Calculate class balance
    total_samples = len(data)
    for class_label, count in target_dist.items():
        percentage = (count / total_samples) * 100
        content.append(f"  - Class {class_label}: {count} samples ({percentage:.1f}%)")

    print(f"\n📊 Dataset insights:")
    print(f"   • Total samples: {len(data)}")
    print(f"   • Features: {len(data.columns) - 1}")
    print(f"   • Target distribution: {target_dist}")

    content.append("")
    return content