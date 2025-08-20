from core.kernels import create_named_kernels

def get_parameter_grids():
    """
    Choose different values for different models' hyperparameters.
    To try different grids, just change the values here.
    """
    return {
        'lr': {
            'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.12, 0.15],
            'regularization_strength': [0.005, 0.01, 0.02, 0.05],
            'epochs': [1000, 1200, 1500]
        },
        'svm': {
            'lambda_': [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2],
            'max_iter': [1200, 1500, 2000, 3000, 4000, 5000],
        },
        'klr': {
            "kernel": create_named_kernels(gamma_values=[0.1, 0.15, 0.2], degree_values=[2, 3], coef0_values=[0.5, 1]), #gamma: 0.12, degree: 4, coef0: 1.5
            "lambda_": [0.005, 0.01], # 0.001
            "epochs": [500, 600] # 1000
        },
        'ksvm': {
            "kernel": create_named_kernels(gamma_values=[0.15, 0.2], degree_values=[2, 3], coef0_values=[0.5, 1]), #gamma: 0.1, 0.3 degree: 4, coef0: 1.5
            "lambda_": [0.0005, 0.001, 0.005], #0.01, 0.05
            "max_iter": [2000, 3000], #1000, 1500
        }
    }
