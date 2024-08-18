import copy
import numpy as np
from sklearn.metrics import accuracy_score
import tenseal as ts

class LogisticRegression():
    LR = 0.1
    SIGMOID_POLY = [0.5, 0.197, 0, -0.004]
    def __init__(self):
        self.losses = []
        self.train_accuracies = []

    def fit(self, x, y, epochs):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

        self.weights /= 10000
        self.bias /= 10000

    def fitVerbose(self, x, y, epochs):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for i in range(epochs):
            print(f"Iterations {i+1}")
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            print(f"errow_w: {error_w}")
            print(f"errow_b: {error_b}")
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

        self.weights /= 10000
        self.bias /= 10000

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        # derivative of binary cross entropy
        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b):
        # self.weights = self.weights - 0.1 * error_w
        self.weights = self.weights - (self.LR * error_w + self.weights * self.LR/2)
        self.bias = self.bias - (self.LR * error_b + self.bias * self.LR/2)

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def predictProb(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return probabilities

    def predictEncrypted(self, x):
        x_dot_weights = [(i.dot(self.weights.transpose()) + self.bias) for i in x]
        enc_probabilities = [(i.polyval(self.SIGMOID_POLY)) for i in x_dot_weights]
        return enc_probabilities

    def predictSingle(self, x):
        x_dot_weight = np.matmul(x, self.weights.transpose()) + self.bias
        probability = self._sigmoid_function(x_dot_weight)
        return 1 if probability > 0.5 else 0

    def predictSingleProb(self, x):
        x_dot_weight = np.matmul(x, self.weights.transpose()) + self.bias
        probability = self._sigmoid_function(x_dot_weight)
        return probability

    def predictEncryptedSingle(self, x):
        enc_x_dot_weight = x.dot(self.weights.transpose()) + self.bias
        enc_probability = enc_x_dot_weight.polyval(self.SIGMOID_POLY)
        return enc_probability

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        if isinstance(x, np.ndarray):
            return x
        else:
            return x.values

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        if isinstance(y, list):
            y = np.array(y)
        if isinstance(y, np.ndarray):
            return y.reshape(y.shape[0], 1)
        else:
            return y.values.reshape(y.shape[0], 1)