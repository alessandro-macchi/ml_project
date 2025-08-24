import os
import random
import numpy as np
from data_processing import load_and_combine_data
from training import run_experiment
from evaluation import generate_final_summary_report
from utils import reset_directory_manager


def main():
    """
    Main function orchestrating the entire ML pipeline
    """

    random_seed = 6
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Reset and initialize the centralized directory manager for this run
    print("📁 Initializing centralized directory management...")
    reset_directory_manager()

    # Load data
    red_path = os.path.join("data", "winequality-red.csv")
    white_path = os.path.join("data", "winequality-white.csv")
    data = load_and_combine_data(red_path, white_path)

    print("\n🚀 Starting classification...")

    # Run experiment
    results, trained_models, overfitting_analyzer = run_experiment(
        data,
        experiment_name="Wine Classification"
    )

    # Generate the final comprehensive report
    generate_final_summary_report(results, overfitting_analyzer, data)

    return results, trained_models, overfitting_analyzer


if __name__ == "__main__":
    main()