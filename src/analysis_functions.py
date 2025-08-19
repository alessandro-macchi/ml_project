import os
from datetime import datetime
from src.overfitting_analysis import integrate_overfitting_analysis
from src.save import save_results, get_directory_manager
from src.utils import get_model_names
from src.visualization import create_model_visualizations


def run_comprehensive_analysis(results, trained_models, X_train, y_train, X_test, y_test, experiment_name):
    """
    Run comprehensive analysis including visualizations and overfitting detection

    Args:
        results: Model performance results
        trained_models: Dictionary of trained model objects
        X_train, y_train: Training data
        X_test, y_test: Test data
        experiment_name: Name of the experiment

    Returns:
        overfitting_analyzer: Overfitting analysis results
    """

    # Save results to results directory (uses centralized directory management)
    save_results(results, experiment_name)

    # Standard visualizations - save to evaluation_plots
    run_visualizations(results, trained_models, X_test, y_test)

    # Overfitting analysis - with consistent directory usage
    overfitting_analyzer = run_overfitting_analysis(trained_models, X_train, y_train, X_test, y_test)

    return overfitting_analyzer


def run_visualizations(results, trained_models, X_test, y_test):
    """Generate standard performance visualizations using centralized directory management"""
    print(f"\n{'=' * 70}")
    print("🎨 GENERATING PERFORMANCE VISUALIZATIONS")
    print(f"{'=' * 70}")

    # Use the streamlined visualization function
    # This will automatically use the centralized directory manager
    visualizer = create_model_visualizations(
        trained_models,
        results,
        X_test,
        y_test,
        model_names=get_model_names(),
        save_plots=True,
        save_dir=None  # Use centralized directory manager
    )
    return visualizer


def run_overfitting_analysis(trained_models, X_train, y_train, X_test, y_test):
    """Run overfitting/underfitting analysis with consistent directory structure"""
    print(f"\n{'=' * 70}")
    print("🎯 OVERFITTING/UNDERFITTING ANALYSIS")
    print(f"{'=' * 70}")

    model_names = get_model_names()

    # Use the integrated overfitting analysis function
    # This will automatically use centralized directory management
    overfitting_analyzer = integrate_overfitting_analysis(
        trained_models,
        X_train,
        y_train,
        X_test,
        y_test,
        model_names,
        save_plots=True,
        save_dir=None  # Use centralized directory manager
    )

    return overfitting_analyzer


def generate_final_summary_report(results, overfitting_analyzer, data):
    """Generate comprehensive final summary report and save to file using centralized directory"""
    print(f"\n{'=' * 80}")
    print("📋 FINAL SUMMARY REPORT")
    print(f"{'=' * 80}")

    model_names = get_model_names()

    # Use centralized directory manager for results
    try:
        dir_manager = get_directory_manager()
        output_dir = dir_manager.results_dir
    except:
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

    report_content.append("")
    report_content.append("🎯 For detailed analysis, refer to the generated visualizations and CSV reports.")
    report_content.append("✅ All analyses complete!")

    # Save report to file
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))

        print(f"\n💾 Final summary report saved to: {report_path}")
    except Exception as e:
        print(f"⚠️ Could not save report to file: {e}")

    # Print to terminal (original behavior)
    for line in report_content:
        print(line)

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
