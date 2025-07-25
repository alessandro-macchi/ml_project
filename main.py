import os
from src.preprocessing import load_and_combine_data, preprocess_features
from src.logistic_regression import LogisticRegressionScratch
from src.hyperparameter_tuning import grid_search
from src.svm import SVMClassifierScratch
from sklearn.linear_model import LogisticRegression #just for comparison
from sklearn.svm import SVC #just for comparison
from sklearn.metrics import accuracy_score #just for evaluation

def main():
    '''Pre-processing data'''
    # Load data
    red_path = os.path.join("data", "winequality-red.csv")
    white_path = os.path.join("data", "winequality-white.csv")
    data = load_and_combine_data(red_path, white_path)

    # Preprocess features and split into train/test
    X_train, X_test, y_train, y_test = preprocess_features(data)


    '''Logistic Regression'''
    # Hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1]
    epoch_list = [100, 500, 1000]
    reg_list = [0, 0.01, 0.1]

    # Grid search
    print("🔍 Starting hyperparameter tuning...")
    best_params_lr, best_score_lr = grid_search(X_train, y_train, LogisticRegressionScratch, learning_rates, epoch_list, reg_list)

    print(f"✅ Best hyperparameters: {best_params_lr}, CV Accuracy: {best_score_lr:.4f}")

    # Final model training
    model_lr_scratch = LogisticRegressionScratch(learning_rate=best_params_lr["learning_rate"], regularization_strength=best_params_lr["regularization_strength"])
    model_lr_scratch.fit(X_train, y_train, epochs=best_params_lr["epochs"])
    pred_lr_scratch = model_lr_scratch.predict(X_test)
    accuracy_lr_scratch = accuracy_score(y_test, pred_lr_scratch)

    # Compare it with the sklearn model
    model_lr_sk = LogisticRegression(penalty=None, solver='lbfgs', max_iter=150)
    model_lr_sk.fit(X_train, y_train)
    pred_lr_sk = model_lr_sk.predict(X_test)
    accuracy_lr_sk = accuracy_score(y_test, pred_lr_sk)
    print(f"✅ Logistic Regression from Scratch has an accuracy of: {accuracy_lr_scratch}")
    print(f"✅ Logistic Regression from sklearn has an accuracy of: {accuracy_lr_sk}")


    '''Support Vector Machine'''
    print("\n🔍 Starting SVM hyperparameter tuning...")
    best_params_svm, best_score_svm = grid_search(
        X_train, y_train, SVMClassifierScratch,
        learning_rates, epoch_list, reg_list
    )

    print(f"✅ Best SVM hyperparameters: {best_params_svm}, CV Accuracy: {best_score_svm:.4f}")

    # Train final model with best params
    svm_model = SVMClassifierScratch(
        learning_rate=best_params_svm["learning_rate"],
        regularization_strength=best_params_svm["regularization_strength"]
    )
    svm_model.fit(X_train, y_train, epochs=best_params_svm["epochs"])
    svm_preds = svm_model.predict(X_test)
    accuracy_svm_scratch = accuracy_score(y_test, svm_preds)

    # Compare it with sklearn's SVM
    print("🔍 Evaluating sklearn's SVM...")
    svm_sk = SVC(kernel='linear', C=1.0, max_iter=-1)
    svm_sk.fit(X_train, y_train)
    pred_sk = svm_sk.predict(X_test)
    accuracy_svm_sk = accuracy_score(y_test, pred_sk)

    print(f"✅ SVM from scratch accuracy: {accuracy_svm_scratch}")
    print(f"✅ SVM from sklearn accuracy: {accuracy_svm_sk}")

if __name__ == "__main__":
    main()
