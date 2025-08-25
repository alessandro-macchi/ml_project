from utils import save_results, get_model_names
from .visualization.visualizer import create_model_visualizations
from .visualization.plots.learning_curves import integrate_overfitting_analysis


def run_comprehensive_analysis(results, trained_models, X_train, y_train, X_test, y_test, experiment_name):

    save_results(results, experiment_name)

    run_visualizations(results, trained_models, X_test, y_test)

    overfitting_analyzer = run_overfitting_analysis(trained_models, X_train, y_train, X_test, y_test)

    return overfitting_analyzer


def run_visualizations(results, trained_models, X_test, y_test):
    print(f"\n{'=' * 70}")
    print("🎨 Generating Performance Visualizations")
    print(f"{'=' * 70}")

    visualizer = create_model_visualizations(
        trained_models,
        results,
        X_test,
        y_test,
        model_names=get_model_names(),
        save_plots=True,
        save_dir=None
    )
    return visualizer


def run_overfitting_analysis(trained_models, X_train, y_train, X_test, y_test):
    """Run overfitting/underfitting analysis with consistent directory structure"""
    print(f"\n{'=' * 70}")
    print("🎯 Overfitting/Underfitting Analysis")
    print(f"{'=' * 70}")

    model_names = get_model_names()

    overfitting_analyzer = integrate_overfitting_analysis(
        trained_models,
        X_train,
        y_train,
        X_test,
        y_test,
        model_names,
        save_plots=True,
        save_dir=None
    )

    return overfitting_analyzer