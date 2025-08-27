
def get_parameter_grids():
    """
    Choose different values for different models' hyperparameters.
    To try different grids, just change the values here.
    The commented values have been tried and excluded for computational time.
    """
    return {
        'logistic_regression': {
            'learning_rate': [0.01, 0.05, 0.1], # 0.01, 0.05, 0.08, 0.1, 0.12, 0.15
            'regularization_strength': [0.02, 0.05, 0.08, 0.1], # 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2
            'epochs': [150, 500, 750, 1000, 1500] # 150, 300, 400, 500, 750, 1000, 1200, 1500
        },
        'svm': {
            'lambda_': [0.01, 0.02, 0.04, 0.06, 0.1, 0.15, 0.2], # 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2
            'max_iter': [1500, 2000, 3000, 4000], #150, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000
        },
        'kernel_logistic_regression': {
            "gamma_values": [0.1, 0.15, 0.2], # 0.1, 0.12, 0.15, 0.2
            "degree_values": [2], # 2, 3, 4
            "coef0_values": [1], # 0.5, 1, 1.5
            "lambda_": [0.005, 0.01], # 0.001, 0.005, 0.01
            "epochs": [300, 500, 1000] # 150, 300, 500, 600, 1000
        },
        'kernel_svm': {
            "gamma_values": [0.2, 0.3], # 0.1, 0.15, 0.2, 0.3
            "degree_values": [2], # 2, 3, 4
            "coef0_values": [1], # 0.5, 1, 1.5
            "lambda_": [0.005, 0.01], # 0.0005, 0.001, 0.005, 0.01, 0.05
            "max_iter": [2000, 3000], # 150, 300, 500, 750, 1000, 1500
        }
    }
