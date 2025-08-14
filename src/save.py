import os
import json
import pickle
from datetime import datetime


def save_results(results, experiment_name, output_dir="output"):
    """Save experiment results into a date-based directory structure."""

    # Directory name is YYYYMMDD
    date_dir = datetime.now().strftime("%Y%m%d")
    base_dir = os.path.join(output_dir, date_dir)

    # Create subdirectories
    results_dir = os.path.join(base_dir, "results")
    plots_dir = os.path.join(base_dir, "evaluation_plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Timestamp for filenames (includes time)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON filename
    json_filename = f"{experiment_name.lower().replace(' ', '_')}_{timestamp}.json"
    json_path = os.path.join(results_dir, json_filename)

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
            'timestamp': timestamp
        }

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    # Pickle filename
    pickle_filename = f"{experiment_name.lower().replace(' ', '_')}_{timestamp}.pkl"
    pickle_path = os.path.join(results_dir, pickle_filename)

    # Save Pickle
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'experiment_name': experiment_name,
            'timestamp': timestamp
        }, f)

    print(f"💾 Results saved in: {results_dir}")
    print(f"   JSON: {json_path}")
    print(f"   Pickle: {pickle_path}")
    print(f"📊 Plots directory ready at: {plots_dir}")

    return json_path, pickle_path, plots_dir
