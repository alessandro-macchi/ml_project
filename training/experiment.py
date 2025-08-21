from data_processing import preprocess_features
from .model_training import train_all_models
from evaluation import run_comprehensive_analysis


def run_experiment(data, experiment_name=""):

    # 1. DATA PREPROCESSING
    X_train, X_test, y_train, y_test = preprocess_features(data, apply_smote=True)

    print(f"✅ Data preprocessing completed with SMOTE oversampling")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # 2. MODEL TRAINING - To try different grids, check hyperparameter_tuning/parameters_grid.py
    results, trained_models = train_all_models(X_train, y_train, X_test, y_test)

    # 3. EVALUATION AND ANALYSIS
    overfitting_analyzer = run_comprehensive_analysis(
        results, trained_models, X_train, y_train, X_test, y_test, experiment_name
    )

    return results, trained_models, overfitting_analyzer