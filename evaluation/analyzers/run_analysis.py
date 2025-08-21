from utils import save_results, get_model_names
from evaluation.visualizers.plots import create_model_visualizations
from .overfitting import integrate_overfitting_analysis


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



