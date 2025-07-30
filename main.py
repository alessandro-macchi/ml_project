import os
from src.preprocessing import load_and_combine_data, preprocess_features
from src.logistic_regression import LogisticRegressionScratch
from src.svm import SVMClassifierScratch
from src.kernels import KernelLogisticRegression, KernelPegasosSVM, create_named_kernels
from src.hyperparameter_tuning import grid_search
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
    lr_param_grid = {
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5],
        "regularization_strength": [0.0, 0.01, 0.1, 0.5],
        "epochs": [500, 1000, 2000]
    }

    print("🔍 Grid search for Logistic Regression...")
    best_params_lr, best_score_lr = grid_search(X_train, y_train, LogisticRegressionScratch, lr_param_grid)
    print(f"✅ Best Logistic Regression params: {best_params_lr}, CV Accuracy: {best_score_lr:.4f}")

    model_lr = LogisticRegressionScratch(
        learning_rate=best_params_lr["learning_rate"],
        regularization_strength=best_params_lr["regularization_strength"],
        epochs=best_params_lr["epochs"]
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

    # Create kernel functions
    gamma_values = [0.001, 0.01, 0.1]
    degree_values = [] #2, 3, 4
    coef0_values = [] #0, 1, 10
    named_kernels = create_named_kernels(gamma_values, degree_values, coef0_values)

    # Parameter grid for kernel logistic regression
    klr_param_grid = {
        "kernel": named_kernels,
        "lambda_": [0.01, 0.1, 0.5],
        "epochs": [500, 1000]
    }

    best_params_klr, best_score_klr = grid_search(X_train, y_train, KernelLogisticRegression, klr_param_grid)
    print(f"🏆 Best Kernel Logistic Regression Params: {best_params_klr}, CV Accuracy: {best_score_klr:.4f}")

    # Train the best model
    klr = KernelLogisticRegression(
        kernel=best_params_klr["kernel"],
        lambda_=best_params_klr["lambda_"],
        epochs=best_params_klr["epochs"],
        subsample_ratio=0.2,  # Use only 20% of training data as support vectors
        batch_size=64,        # Reasonable batch size
        early_stopping_patience=20
    )

    klr.fit(X_train, y_train)
    klr_preds = klr.predict(X_test)
    acc_klr = accuracy_score(y_test, klr_preds)

    print(f"✅ Kernel Logistic Regression from Scratch Accuracy: {acc_klr:.4f}")

    '''Linear SVM'''
    print("\n🔍 Starting Linear SVM hyperparameter tuning...")

    svm_param_grid = {
        "lambda_": [0.001, 0.01, 0.1],
        "max_iter": [100, 500, 1000]
    }

    best_params_svm, best_score_svm = grid_search(X_train, y_train, SVMClassifierScratch, svm_param_grid)
    print(f"✅ Best Linear SVM params: {best_params_svm}, CV Accuracy: {best_score_svm:.4f}")

    svm_model = SVMClassifierScratch(lambda_=best_params_svm["lambda_"])
    svm_model.fit(X_train, y_train, max_iter=best_params_svm["max_iter"])
    acc_svm = accuracy_score(y_test, svm_model.predict(X_test))

    l_svm_sk = SVC(kernel='linear', C=1.0)
    l_svm_sk.fit(X_train, y_train)
    acc_l_svm_sk = accuracy_score(y_test, l_svm_sk.predict(X_test))

    print(f"✅ Linear SVM from Scratch Accuracy: {acc_svm:.4f}")
    print(f"✅ Linear SVM (sklearn) Accuracy: {acc_l_svm_sk:.4f}")

    '''Kernel SVM'''
    print("\n🔍 Starting Kernel SVM hyperparameter tuning...")
    gamma_values = [0.001, 0.01, 0.1]
    degree_values = [2, 3, 4]
    coef0_values = [0, 1, 10]
    named_kernels = create_named_kernels(gamma_values, degree_values, coef0_values)

    ksvm_param_grid = {
        "kernel_fn": named_kernels,
        "lambda_": [0.01, 0.1],
        "max_iter": [500, 1000]
    }

    best_params_ksvm, best_score_ksvm = grid_search(X_train, y_train, KernelPegasosSVM, ksvm_param_grid)
    print(f"🏆 Best Kernel Pegasos Params: {best_params_ksvm}, CV Accuracy: {best_score_ksvm:.4f}")

    ksvm = KernelPegasosSVM(
        kernel_fn=best_params_ksvm["kernel_fn"],
        lambda_=best_params_ksvm["lambda_"],
        max_iter=best_params_ksvm["max_iter"]
    )
    ksvm.fit(X_train, y_train)
    ksvm_preds = ksvm.predict(X_test)
    acc_ksvm = accuracy_score(y_test, ksvm_preds)
    
    print(f"✅ Kernel SVM from Scratch Accuracy: {acc_ksvm:.4f}")
    print(f"📊 Number of support vectors: {len(ksvm.support_vectors)}")

    rbf_svm_sk = SVC(kernel='rbf', C=1.0)
    rbf_svm_sk.fit(X_train, y_train)
    acc_rbf_svm_sk = accuracy_score(y_test, rbf_svm_sk.predict(X_test))

    print(f"✅ Kernel RBF SVM from Scratch Accuracy: {acc_rbf_svm_sk:.4f}")

    poly_svm_sk = SVC(kernel='poly', C=1.0)
    poly_svm_sk.fit(X_train, y_train)
    acc_poly_svm_sk = accuracy_score(y_test, poly_svm_sk.predict(X_test))

    print(f"✅ Kernel Poly SVM from Scratch Accuracy: {acc_poly_svm_sk:.4f}")

    '''Summary'''
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 50)
    print(f"Linear Logistic Regression (Scratch): {acc_lr:.4f}")
    print(f"Linear Logistic Regression (sklearn): {acc_lr_sk:.4f}")
    print(f"Kernel Logistic Regression (Scratch): {acc_klr:.4f}")
    print(f"Linear SVM (Scratch): {acc_svm:.4f}")
    print(f"Linear SVM (sklearn): {acc_l_svm_sk:.4f}")
    print(f"Kernel SVM (Scratch): {acc_ksvm:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()