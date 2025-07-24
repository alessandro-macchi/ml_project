import os
from src.preprocessing import load_and_combine_data, preprocess_features

def main():
    # Load data
    red_path = os.path.join("data", "winequality-red.csv")
    white_path = os.path.join("data", "winequality-white.csv")
    data = load_and_combine_data(red_path, white_path)

    # Preprocess features and split into train/test
    X_train, X_test, y_train, y_test = preprocess_features(data)



if __name__ == "__main__":
    main()
