import numpy as np
from sklearn.metrics import accuracy_score  #just for evaluation

class LogisticRegressionScratch:
    def __init__(self, learning_rate, regularization_strength):
        self.weights = 0
        self.bias = 0
        self.learning_rate = learning_rate
        self.lambda_ = regularization_strength
        self.losses = []
        self.train_accuracies = []

    def fit(self, x, y, epochs):
        self.weights = np.zeros(x.shape[1])
        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def compute_loss(self, y_true, y_pred):
        # Binary cross entropy + L2 regularization term
        bce = -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))
        l2_penalty = (self.lambda_ / 2) * np.sum(self.weights ** 2)
        return bce + l2_penalty

    def compute_gradients(self, x, y_true, y_pred):
        # Gradient of loss + L2 penalty
        diff = y_pred - y_true
        gradient_b = np.mean(diff)
        gradients_w = np.dot(x.T, diff) / len(y_true)
        gradients_w += self.lambda_ * self.weights  # L2 regularization gradient
        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - self.learning_rate * error_w
        self.bias = self.bias - self.learning_rate * error_b

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]



