import os
from src.preprocessing import load_and_combine_data, preprocess_features
from models.logistic_regression import run_logistic_regression_experiment
from models.svm import run_svm_experiment
from models.kernel_logistic_regression import run_kernel_logistic_regression_experiment
from models.kernel_svm import run_kernel_svm_experiment
from src.utils import create_named_kernels
from src.save import save_results
from src.visualization import integrate_with_experiment_results
from src.misclassification_analysis import MisclassificationAnalyzer
from src.hyperparameter_tuning import grid_search
from src.overfitting_analysis import integrate_overfitting_analysis


def run_logistic_regression_experiment_with_model(X_train, y_train, X_test, y_test, param_grid):
    """Modified to return both results and trained model"""
    from models.logistic_regression import LogisticRegressionScratch

    print("🔍 Grid search for Logistic Regression...")
    best_params, best_score = grid_search(X_train, y_train, LogisticRegressionScratch, param_grid)
    print(f"✅ Best LR params: {best_params}, CV Accuracy: {best_score:.4f}")

    model = LogisticRegressionScratch(**best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    from src.metrics import comprehensive_evaluation
    results = {'lr_custom': comprehensive_evaluation(y_test, preds, "Logistic Regression (Custom)")}

    return results, model


def run_svm_experiment_with_model(X_train, y_train, X_test, y_test, param_grid):
    """Modified to return both results and trained model"""
    from models.svm import SVMClassifierScratch

    print("🔍 Grid search for Linear SVM...")
    best_params, best_score = grid_search(X_train, y_train, SVMClassifierScratch, param_grid)
    print(f"✅ Best SVM params: {best_params}, CV Accuracy: {best_score:.4f}")

    model = SVMClassifierScratch(lambda_=best_params["lambda_"])
    model.fit(X_train, y_train, max_iter=best_params["max_iter"])
    preds = model.predict(X_test)

    from src.metrics import comprehensive_evaluation
    results = {'svm_custom': comprehensive_evaluation(y_test, preds, "Linear SVM (Custom)")}

    return results, model


def run_kernel_logistic_regression_experiment_with_model(X_train, y_train, X_test, y_test, param_grid):
    """Modified to return both results and trained model"""
    from models.kernel_logistic_regression import KernelLogisticRegression

    print("🔍 Grid search for Kernel Logistic Regression...")
    best_params, best_score = grid_search(X_train, y_train, KernelLogisticRegression, param_grid)
    print(f"✅ Best KLR params: {best_params}, CV Accuracy: {best_score:.4f}")

    model = KernelLogisticRegression(
        kernel=best_params["kernel"],
        lambda_=best_params["lambda_"],
        epochs=best_params["epochs"],
        subsample_ratio=0.2,
        batch_size=64,
        early_stopping_patience=20
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    from src.metrics import comprehensive_evaluation
    results = {'klr_custom': comprehensive_evaluation(y_test, preds, "Kernel Logistic Regression (Custom)")}

    return results, model


def run_kernel_svm_experiment_with_model(X_train, y_train, X_test, y_test, param_grid):
    """Modified to return both results and trained model"""
    from models.kernel_svm import KernelPegasosSVM

    print("🔍 Grid search for Kernel SVM (Pegasos)...")
    best_params, best_score = grid_search(X_train, y_train, KernelPegasosSVM, param_grid)
    print(f"✅ Best KSVM params: {best_params}, CV Accuracy: {best_score:.4f}")

    model = KernelPegasosSVM(
        kernel=best_params["kernel"],
        lambda_=best_params["lambda_"],
        max_iter=best_params["max_iter"]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"📊 Number of support vectors: {len(model.support_vectors)}")

    from src.metrics import comprehensive_evaluation
    results = {'ksvm_custom': comprehensive_evaluation(y_test, preds, "Kernel SVM (Pegasos)")}

    return results, model


def run_experiment(data, experiment_name=""):
    """
    Enhanced version that includes comprehensive analysis including overfitting detection
    """
    print(f"\n{'=' * 70}")
    print(f"🧪 EXPERIMENT WITH COMPREHENSIVE ANALYSIS: {experiment_name}")
    print(f"{'=' * 70}")

    # Apply SMOTE only to Train, to avoid Data Leakage
    X_train, X_test, y_train, y_test = preprocess_features(data, apply_smote=True)

    print(f"✅ Data preprocessing completed with SMOTE oversampling")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    results = {}
    trained_models = {}

    # LOGISTIC REGRESSION
    print(f"\n🔹 Running Logistic Regression...")
    lr_param_grid = {
        'learning_rate': [0.08, 0.1, 0.12, 0.15],
        'regularization_strength': [0.005, 0.01, 0.02, 0.05],
        'epochs': [1000, 1200, 1500]
    }

    lr_results, lr_model = run_logistic_regression_experiment_with_model(X_train, y_train, X_test, y_test,
                                                                         lr_param_grid)
    results.update(lr_results)
    trained_models['lr_custom'] = lr_model

    # SVM
    print(f"\n🔹 Running SVM...")
    svm_param_grid = {
        'lambda_': [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2],
        'max_iter': [1200, 1500, 2000, 3000],
    }

    svm_results, svm_model = run_svm_experiment_with_model(X_train, y_train, X_test, y_test, svm_param_grid)
    results.update(svm_results)
    trained_models['svm_custom'] = svm_model

    # KERNEL LOGISTIC REGRESSION
    print(f"\n🔹 Running Kernel Logistic Regression...")
    klr_param_grid = {
        "kernel": create_named_kernels(gamma_values=[0.1, 0.12, 0.15], degree_values=[], coef0_values=[]),
        "lambda_": [0.005, 0.01],
        "epochs": [500, 600]
    }

    klr_results, klr_model = run_kernel_logistic_regression_experiment_with_model(X_train, y_train, X_test, y_test,
                                                                                  klr_param_grid)
    results.update(klr_results)
    trained_models['klr_custom'] = klr_model

    # KERNEL SVM
    print(f"\n🔹 Running Kernel SVM...")
    ksvm_param_grid = {
        "kernel": create_named_kernels(gamma_values=[0.1, 0.15], degree_values=[2, 3], coef0_values=[0.5, 1, 1.5]),
        "lambda_": [0.005, 0.01, 0.05],
        "max_iter": [1000, 1500]
    }

    ksvm_results, ksvm_model = run_kernel_svm_experiment_with_model(X_train, y_train, X_test, y_test, ksvm_param_grid)
    results.update(ksvm_results)
    trained_models['ksvm_custom'] = ksvm_model

    # Print results
    print_model_results(results, experiment_name)

    # Save results
    save_results(results, experiment_name)

    # STANDARD VISUALIZATIONS
    print(f"\n{'=' * 70}")
    print("🎨 GENERATING PERFORMANCE VISUALIZATIONS")
    print(f"{'=' * 70}")

    visualizer = integrate_with_experiment_results(results, X_test, y_test)

    # ENHANCED MISCLASSIFICATION ANALYSIS
    print(f"\n{'=' * 70}")
    print("🔍 ENHANCED MISCLASSIFICATION ANALYSIS")
    print(f"{'=' * 70}")

    wine_feature_names = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'wine_type'
    ]

    model_names = {
        'lr_custom': 'Logistic Regression (Custom)',
        'svm_custom': 'Linear SVM (Custom)',
        'klr_custom': 'Kernel Logistic Regression (Custom)',
        'ksvm_custom': 'Kernel SVM (Custom)'
    }

    analyzer = MisclassificationAnalyzer()
    analyzer.analyze_all_models(trained_models, X_test, y_test, wine_feature_names, model_names)
    analyzer.create_comprehensive_analysis()
    analyzer.export_analysis_results("wine_misclassification_analysis.csv")

    # NEW: OVERFITTING/UNDERFITTING ANALYSIS
    print(f"\n{'=' * 70}")
    print("🎯 OVERFITTING/UNDERFITTING ANALYSIS")
    print(f"{'=' * 70}")

    overfitting_analyzer = integrate_overfitting_analysis(
        trained_models, X_train, y_train, X_test, y_test, model_names
    )

    return results, trained_models, analyzer, overfitting_analyzer


def print_model_results(results, experiment_name):
    """Print formatted results for all models"""
    print(f"\n{'=' * 70}")
    print(f"📊 RESULTS SUMMARY: {experiment_name}")
    print(f"{'=' * 70}")

    for model_name, metrics in results.items():
        print(f"\n🔹 {model_name}:")

        if isinstance(metrics, dict) and 'accuracy' in metrics:
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"   Balanced Accuracy: {metrics.get('balanced_accuracy', 'N/A'):.4f}")
            print(f"   Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"   Recall: {metrics.get('recall', 'N/A'):.4f}")
            print(f"   F1-Score: {metrics.get('f1', 'N/A'):.4f}")
        else:
            print(f"   Raw result: {metrics}")


def main():
    """
    Enhanced main function with comprehensive analysis including overfitting detection
    """
    print("🍷 WINE QUALITY CLASSIFICATION WITH COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    red_path = os.path.join("data", "winequality-red.csv")
    white_path = os.path.join("data", "winequality-white.csv")

    data = load_and_combine_data(red_path, white_path)

    print("\n🚀 Starting experiment with comprehensive analysis...")

    results, trained_models, misclassification_analyzer, overfitting_analyzer = run_experiment(
        data,
        experiment_name="Wine Classification with SMOTE and Comprehensive Analysis"
    )

    print(f"\n{'=' * 80}")
    print("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("📊 Check the visualizations and reports above for detailed insights.")
    print("💾 Analysis results saved to:")
    print("   • wine_misclassification_analysis.csv")
    print("   • overfitting_analysis.csv")
    print(f"{'=' * 80}")

    # Generate final summary report
    print(f"\n{'=' * 80}")
    print("📋 FINAL SUMMARY REPORT")
    print(f"{'=' * 80}")

    # Best performing model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"🏆 Best performing model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

    # Model names for display
    model_names = {
        'lr_custom': 'Logistic Regression (Custom)',
        'svm_custom': 'Linear SVM (Custom)',
        'klr_custom': 'Kernel Logistic Regression (Custom)',
        'ksvm_custom': 'Kernel SVM (Custom)'
    }

    # Overfitting summary
    overfitting_summary = {}
    for model_key, analysis in overfitting_analyzer.analysis_results.items():
        status = analysis['fitting_diagnosis']['fitting_status']
        overfitting_summary[model_key] = status

    print(f"🎯 Overfitting status summary:")
    for model_key, status in overfitting_summary.items():
        emoji = {'overfitting': '🔴', 'underfitting': '🟡', 'good_fit': '🟢', 'inconclusive': '⚪'}.get(status, '⚪')
        model_name = model_names.get(model_key, model_key)
        print(f"   {emoji} {model_name}: {status.upper()}")

    # Performance summary
    print(f"\n📈 Performance summary:")
    for model_key, metrics in results.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            model_name = model_names.get(model_key, model_key)
            accuracy = metrics['accuracy']
            f1 = metrics['f1']
            print(f"   📊 {model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

    # General recommendations
    print(f"\n💡 FINAL RECOMMENDATIONS:")

    overfitting_models = [k for k, v in overfitting_summary.items() if v == 'overfitting']
    underfitting_models = [k for k, v in overfitting_summary.items() if v == 'underfitting']
    good_fit_models = [k for k, v in overfitting_summary.items() if v == 'good_fit']

    if good_fit_models:
        print(f"✅ Well-fitted models: {', '.join([model_names.get(k, k) for k in good_fit_models])}")
        print("   → These models are ready for production use")

    if overfitting_models:
        print(f"🔴 Overfitting detected in: {', '.join([model_names.get(k, k) for k in overfitting_models])}")
        print("   → Consider: increased regularization, more data, or simpler models")

    if underfitting_models:
        print(f"🟡 Underfitting detected in: {', '.join([model_names.get(k, k) for k in underfitting_models])}")
        print("   → Consider: reduced regularization, more complex models, or feature engineering")

    # Data insights
    print(f"\n📊 Dataset insights:")
    print(f"   • Total samples: {len(data)}")
    print(f"   • Features: {len(data.columns) - 1}")  # -1 for target variable
    print(f"   • Target distribution: {data['quality_binary'].value_counts().to_dict()}")

    print(f"\n🎯 For detailed analysis, refer to the generated visualizations and CSV reports.")
    print(f"✅ All analyses complete!")

    return results, trained_models, misclassification_analyzer, overfitting_analyzer


if __name__ == "__main__":
    main()