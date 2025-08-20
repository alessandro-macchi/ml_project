"""
Fixed directory management system for consistent timestamp-based organization
"""

import os
import json
import pickle
from datetime import datetime


class DirectoryManager:
    """Centralized directory management for consistent file organization"""

    def __init__(self, base_output_dir="output"):
        self.base_output_dir = base_output_dir
        self.date_dir = datetime.now().strftime("%Y%m%d")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the main directory structure
        self.base_dir = os.path.join(self.base_output_dir, self.date_dir)
        self.results_dir = os.path.join(self.base_dir, "results")
        self.plots_dir = os.path.join(self.base_dir, "evaluation_plots")

        # Create all directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        print(f"📁 Directory structure created:")
        print(f"   Base: {self.base_dir}")
        print(f"   Results: {self.results_dir}")
        print(f"   Plots: {self.plots_dir}")

    def get_results_path(self, filename):
        """Get full path for results file"""
        return os.path.join(self.results_dir, filename)

    def get_plots_path(self, filename):
        """Get full path for plots file"""
        return os.path.join(self.plots_dir, filename)

    def get_timestamped_filename(self, base_name, extension):
        """Generate timestamped filename"""
        name_without_ext = base_name.rsplit('.', 1)[0] if '.' in base_name else base_name
        return f"{name_without_ext}_{self.timestamp}.{extension}"


# Global directory manager instance
_dir_manager = None


def get_directory_manager():
    """Get or create the global directory manager"""
    global _dir_manager
    if _dir_manager is None:
        _dir_manager = DirectoryManager()
    return _dir_manager


def reset_directory_manager():
    """Reset the directory manager (for new runs)"""
    global _dir_manager
    _dir_manager = DirectoryManager()
    return _dir_manager


def save_results(results, experiment_name, output_dir="output"):
    """Save experiment results into a timestamp-based directory structure."""

    # Get or create directory manager
    dir_manager = get_directory_manager()

    # JSON filename with timestamp
    json_filename = dir_manager.get_timestamped_filename(
        f"{experiment_name.lower().replace(' ', '_')}", "json"
    )
    json_path = dir_manager.get_results_path(json_filename)

    # Make results JSON-serializable
    json_results = {}
    for model_name, metrics in results.items():
        json_results[model_name] = {
            'accuracy': float(metrics.get('accuracy', 0.0)),
            'balanced_accuracy': float(metrics.get('balanced_accuracy', metrics.get('accuracy', 0.0))),
            'precision': float(metrics.get('precision', 0.0)),
            'recall': float(metrics.get('recall', 0.0)),
            'f1': float(metrics.get('f1', 0.0)),
            'experiment': experiment_name,
            'timestamp': dir_manager.timestamp
        }

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    # Pickle filename with timestamp
    pickle_filename = dir_manager.get_timestamped_filename(
        f"{experiment_name.lower().replace(' ', '_')}", "pkl"
    )
    pickle_path = dir_manager.get_results_path(pickle_filename)

    # Save Pickle
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'experiment_name': experiment_name,
            'timestamp': dir_manager.timestamp
        }, f)

    print(f"💾 Results saved in: {dir_manager.results_dir}")
    print(f"   JSON: {json_path}")
    print(f"   Pickle: {pickle_path}")
    print(f"📊 Plots directory ready at: {dir_manager.plots_dir}")

    return json_path, pickle_path, dir_manager.plots_dir