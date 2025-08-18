import os
from src.preprocessing import load_and_combine_data, preprocess_features
from models.model_training import train_all_models, print_model_results
from src.analysis_functions import run_comprehensive_analysis, generate_final_summary_report
from src.save import reset_directory_manager


def run_experiment(data, experiment_name=""):

    print(f"\n{'=' * 70}")
    print(f"🧪 EXPERIMENT: {experiment_name}")
    print(f"{'=' * 70}")

    # 1. DATA PREPROCESSING
    X_train, X_test, y_train, y_test = preprocess_features(data, apply_smote=True)

    print(f"✅ Data preprocessing completed with SMOTE oversampling")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # 2. MODEL TRAINING - To try different grids, check models/model_training.py
    results, trained_models = train_all_models(X_train, y_train, X_test, y_test)

    # Print initial results
    print_model_results(results, experiment_name)

    # 3. EVALUATION AND ANALYSIS
    misclassification_analyzer, overfitting_analyzer = run_comprehensive_analysis(
        results, trained_models, X_train, y_train, X_test, y_test, experiment_name
    )

    return results, trained_models, misclassification_analyzer, overfitting_analyzer


def main():
    """
    Main function orchestrating the entire ML pipeline
    """
    print("🍷 WINE QUALITY CLASSIFICATION WITH COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # Reset and initialize the centralized directory manager for this run
    print("📁 Initializing centralized directory management...")
    reset_directory_manager()

    # Load data
    red_path = os.path.join("data", "winequality-red.csv")
    white_path = os.path.join("data", "winequality-white.csv")
    data = load_and_combine_data(red_path, white_path)

    print("\n🚀 Starting experiment with comprehensive analysis...")

    # Run experiment
    results, trained_models, misclassification_analyzer, overfitting_analyzer = run_experiment(
        data,
        experiment_name="Wine Classification"
    )

    # Generate the final comprehensive report
    generate_final_summary_report(results, overfitting_analyzer, data)

    return results, trained_models, misclassification_analyzer, overfitting_analyzer


if __name__ == "__main__":
    main()