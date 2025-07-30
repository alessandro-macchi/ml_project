import os
from src.preprocessing import load_and_combine_data, preprocess_features
from src.kernels import KernelPegasosSVM, create_named_kernels
from src.hyperparameter_tuning import grid_search
from sklearn.svm import SVC  # benchmark
from sklearn.metrics import accuracy_score  # evaluation


def main():
    '''Pre-processing data'''
    red_path = os.path.join("data", "winequality-red.csv")
    white_path = os.path.join("data", "winequality-white.csv")
    data = load_and_combine_data(red_path, white_path)
    X_train, X_test, y_train, y_test = preprocess_features(data)
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
