import os
import json
import pickle
from datetime import datetime

def save_results(results, experiment_name, results_dir="output/results"):
    """Save experiment results to files"""
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON (for easy reading)
    json_filename = f"{experiment_name.lower().replace(' ', '_')}_{timestamp}.json"
    json_path = os.path.join(results_dir, json_filename)

    # Convert results to JSON-serializable format
    json_results = {}
    for model_name, metrics in results.items():
        # Handle missing keys gracefully with default values
        json_results[model_name] = {
            'accuracy': float(metrics.get('accuracy', 0.0)),
            'balanced_accuracy': float(metrics.get('balanced_accuracy', metrics.get('accuracy', 0.0))),
            'precision': float(metrics.get('precision', 0.0)),
            'recall': float(metrics.get('recall', 0.0)),
            'f1': float(metrics.get('f1', 0.0)),
            'experiment': experiment_name,
            'timestamp': timestamp
        }

    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    # Save as pickle (for complete object preservation)
    pickle_filename = f"{experiment_name.lower().replace(' ', '_')}_{timestamp}.pkl"
    pickle_path = os.path.join(results_dir, pickle_filename)

    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'experiment_name': experiment_name,
            'timestamp': timestamp
        }, f)

    print(f"💾 Results saved:")
    print(f"   JSON: {json_path}")
    print(f"   Pickle: {pickle_path}")

    return json_path, pickle_path
