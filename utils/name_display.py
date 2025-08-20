def get_model_names():
    """
    Get display names for all models

    Returns:
        dict: Mapping of model keys to display names
    """
    return {
        'lr_custom': 'Logistic Regression',
        'svm_custom': 'Linear SVM',
        'klr_custom': 'Kernel Logistic Regression',
        'ksvm_custom': 'Kernel SVM'
    }