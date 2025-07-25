import os
from src.preprocessing import load_and_combine_data, preprocess_features
from src.logistic_regression import LogisticRegressionScratch
from src.hyperparameter_tuning import grid_search
from sklearn.linear_model import LogisticRegression #just for evaluation
from sklearn.metrics import accuracy_score #just for evaluation

def main():
    '''Preprocessing'''
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
    best_params, best_score = grid_search(X_train, y_train, LogisticRegressionScratch, learning_rates, epoch_list, reg_list)

    print(f"✅ Best hyperparameters: {best_params}, CV Accuracy: {best_score:.4f}")

    # Final model training
    model = LogisticRegressionScratch(learning_rate=best_params["learning_rate"], regularization_strength=best_params["regularization_strength"])
    model.fit(X_train, y_train, epochs=best_params["epochs"])
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    # Compare it with the sklearn model
    model_sk = LogisticRegression(penalty=None, solver='lbfgs', max_iter=150)
    model_sk.fit(X_train, y_train)
    pred2 = model_sk.predict(X_test)
    accuracy2 = accuracy_score(y_test, pred2)
    print(f"✅ Logistic Regression from Scratch has an accuracy of: {accuracy}")
    print(f"✅ Logistic Regression from sklearn has an accuracy of: {accuracy2}")

if __name__ == "__main__":
    main()
