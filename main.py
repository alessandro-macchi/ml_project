import os
from src.preprocessing import load_and_combine_data, preprocess_features
from src.logistic_regression import LogisticRegressionScratch
from src.svm import SVMClassifierScratch
from src.kernels import KernelLogisticRegression, KernelPegasosSVM, create_named_kernels
from src.hyperparameter_tuning import grid_search, grid_search_svm, grid_search_kernel
from sklearn.linear_model import LogisticRegression  # benchmark
from sklearn.svm import SVC  # benchmark
from sklearn.metrics import accuracy_score  # evaluation

def main():
    '''Pre-processing data'''
    red_path = os.path.join("data", "winequality-red.csv")
    white_path = os.path.join("data", "winequality-white.csv")
    data = load_and_combine_data(red_path, white_path)
    X_train, X_test, y_train, y_test = preprocess_features(data)

    '''Logistic Regression'''
    lr_grid = {
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5],
        "regularization_strength": [0, 0.01, 0.1, 0.5],
        "epochs": [500, 1000, 2000]
    }

    print("🔍 Grid search for Logistic Regression...")
    best_params_lr, best_score_lr = grid_search(X_train, y_train, LogisticRegressionScratch, lr_grid["learning_rate"],
                                    lr_grid["epochs"], lr_grid["regularization_strength"])
    print(f"✅ Best Logistic Regression params: {best_params_lr}, CV Accuracy: {best_score_lr:.4f}")

    model_lr = LogisticRegressionScratch(
        learning_rate=best_params_lr["learning_rate"],
        regularization_strength=best_params_lr["regularization_strength"]
    )
    model_lr.fit(X_train, y_train)
    pred_lr = model_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, pred_lr)

    model_lr_sk = LogisticRegression(penalty=None, solver='lbfgs', max_iter=best_params_lr["epochs"])
    model_lr_sk.fit(X_train, y_train)
    acc_lr_sk = accuracy_score(y_test, model_lr_sk.predict(X_test))

    print(f"✅ Logistic Regression from Scratch Accuracy: {acc_lr:.4f}")
    print(f"✅ Logistic Regression (sklearn) Accuracy: {acc_lr_sk:.4f}")

    '''Kernel Logistic Regression'''
    print("\n🔍 Starting Kernel Logistic Regression hyperparameter tuning...")

    # Create kernel functions (same as for SVM)
    gamma_values = [0.001, 0.01, 0.1]
    degree_values = [2, 3, 4]
    coef0_values = [0, 1, 10]

    named_kernels = create_named_kernels(gamma_values, degree_values, coef0_values)

    # Parameter grid for kernel logistic regression
    klr_param_grid = {
        "kernel_fn": named_kernels,
        "lambda_": [0.01, 0.1, 0.5],
        "max_iter": [500, 1000]
    }

    # Grid search for kernel logistic regression
    best_params_klr, best_score_klr = grid_search_kernel(X_train, y_train, KernelLogisticRegression, klr_param_grid)
    print(f"🏆 Best Kernel Logistic Regression Params: {best_params_klr}, CV Accuracy: {best_score_klr:.4f}")

    # Train the best kernel logistic regression model
    klr = KernelLogisticRegression(
        kernel_fn=best_params_klr["kernel_fn"],
        lambda_=best_params_klr["lambda_"],
        max_iter=best_params_klr["max_iter"]
    )
    klr.fit(X_train, y_train)
    klr_preds = klr.predict(X_test)
    acc_klr = accuracy_score(y_test, klr_preds)

    print(f"✅ Kernel Logistic Regression from Scratch Accuracy: {acc_klr:.4f}")
    print(f"📊 Number of support vectors: {len(klr.support_vectors)}")

    '''Linear SVM'''
    print("\n🔍 Starting Linear SVM hyperparameter tuning...")
    lambda_list = [0.001, 0.01, 0.1]
    max_iter_list = [100, 500, 1000]

    best_params_svm, best_score_svm = grid_search_svm(
        X_train, y_train, SVMClassifierScratch, lambda_list, max_iter_list
    )
    print(f"✅ Best Linear SVM params: {best_params_svm}, CV Accuracy: {best_score_svm:.4f}")

    svm_model = SVMClassifierScratch(lambda_=best_params_svm["lambda_"])
    svm_model.fit(X_train, y_train, epochs=best_params_svm["max_iter"])
    acc_svm = accuracy_score(y_test, svm_model.predict(X_test))

    svm_sk = SVC(kernel='linear', C=1.0)
    svm_sk.fit(X_train, y_train)
    acc_svm_sk = accuracy_score(y_test, svm_sk.predict(X_test))

    print(f"✅ Linear SVM from Scratch Accuracy: {acc_svm:.4f}")
    print(f"✅ Linear SVM (sklearn) Accuracy: {acc_svm_sk:.4f}")

    '''Kernel SVM'''
    gamma_values = [0.001, 0.01, 0.1]
    degree_values = [2, 3, 4]
    coef0_values = [0, 1, 10]

    named_kernels = create_named_kernels(gamma_values, degree_values, coef0_values)

    param_grid = {
        "kernel_fn": named_kernels,
        "lambda_": [0.01, 0.1],
        "max_iter": [500, 1000]
    }

    best_params_ksvm, best_score_ksvm = grid_search_kernel(X_train, y_train, KernelPegasosSVM, param_grid)
    print(f"🏆 Best Kernel Pegasos Params: {best_params_ksvm}, CV Accuracy: {best_score_ksvm:.4f}")

    ksvm = KernelPegasosSVM(kernel_fn=best_params_ksvm["kernel_fn"], lambda_=best_params_ksvm["lambda_"], max_iter=best_params_ksvm["max_iter"])
    ksvm.fit(X_train, y_train)
    ksvm_preds = ksvm.predict(X_test)
    acc_ksvm = accuracy_score(y_test, ksvm_preds)

    print(f"✅ Kernel SVM from Scratch Accuracy: {acc_ksvm:.4f}")


if __name__ == "__main__":
    main()