import numpy as np
import math

# sample dataset
X_train = np.array([-3, -2, -1, 0, 1, 2, 3]).reshape(-1, 1)
y_train = np.array([0, 0, 0, 1, 1, 1, 1])

# sample weights
w = np.array([1])
b = 0

def sigmoid(z: float):
    return 1 / (1 + np.exp(-z))

def binary_crossentropy(y_hat_i: float, y_i: np.ndarray):
    return -y_i * math.log10(y_hat_i) - (1 - y_i) * math.log10(1 - y_hat_i)

def compute_model_output(X: np.ndarray, w: np.ndarray, b: float):
    return np.array([sigmoid(np.dot(X[i], w) + b) for i in range(X.shape[0])])

def compute_cost(y_hat: np.ndarray, y: np.ndarray):
    m = y_hat.shape[0]
    cost = 0.
    for i in range(m):
        cost += binary_crossentropy(y_hat[i], y[i])
    cost /= m
    return cost

predictions = compute_model_output(X_train, w, b)
cost = compute_cost(predictions, y_train)

print("Predictions:", predictions)
print("Cost:", cost)
