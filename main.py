import os
import numpy as np
from data_processing import load_and_combine_data
from training import run_experiment
from evaluation import generate_final_summary_report
from utils import reset_directory_manager


def main():
    np.random.seed(6)

    print("📁 Initializing centralized directory management...")
    reset_directory_manager()

    red_path = os.path.join("data", "winequality-red.csv")
    white_path = os.path.join("data", "winequality-white.csv")
    data = load_and_combine_data(red_path, white_path)

    print("\n🚀 Starting classification...")

    results, trained_models, overfitting_analyzer = run_experiment(data,
                                                        experiment_name="Wine Classification")

    generate_final_summary_report(results, overfitting_analyzer, data)

    return results, trained_models, overfitting_analyzer


if __name__ == "__main__":
    main()