import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-5 # tolerance for imprecision in floating-point calculations

# Initialise dummy house prices dataset
dataset = {
    "size_sqft": [2104., 1416., 852.],
    "bedrooms_count": [5., 3., 2.],
    "floors_count": [1., 2., 1.],
    "property_age_years": [45., 40., 35.],
    "price_1k_usd": [460., 232., 178.]
}

def predict(x: np.ndarray, w: np.ndarray, b: float):
    return np.dot(x, w) + b

def predict_batch(X: np.ndarray, w: np.ndarray, b: float):
    return np.array([predict(X[i], w, b) for i in range(X.shape[0])])

def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    return sum((prediction - y[i]) ** 2 for i, prediction in enumerate(predict_batch(X, w, b))) / (2 * len(y))

def compute_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    m, n = X.shape
    f_wb = predict_batch(X, w, b)

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        err = f_wb[i] - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def gradient_descent(X: np.ndarray, y: np.ndarray, w_init: np.ndarray, b_init: float, alpha: float):
    J_history = []
    w = w_init.copy()
    b = b_init

    i = 0
    while True:
        i += 1
        old_w = w.copy()
        old_b = b

        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        J_history.append(compute_cost(X, y, w, b))

        if sum(abs(old_w - w)) <= EPSILON and abs(old_b - b) <= EPSILON:
            break

    return w, b, J_history

def visualise_gradient_descent(J_history: np.ndarray):
    plt.figure(figsize=(12, 5))
    plt.title("Change in Cost with Iteration")
    plt.xlabel("Iteration Step")
    plt.ylabel("Cost")
    plt.plot(J_history)
    plt.show()

def main(dataset: dict[str, list[float]]):
    target_column = "price_1k_usd"

    n = len(dataset) - 1 # number of features
    m = len(dataset[target_column]) # number of records

    # Note we use "X" instad of "x" because the features are represented as a matrix, not a vector
    X_train = np.array([[dataset[key][i] for key in dataset.keys() if key != target_column] for i in range(m)])
    y_train = np.array(dataset[target_column])

    # Optimise weights
    w, b, J_history = gradient_descent(X_train, y_train, np.zeros(n), 0, 5.0e-7)
    print("Final w weights:", w, "Final b weight:", b, "Cost with final weights:", J_history[-1])
    visualise_gradient_descent(J_history)

    # Predictions with finalised weights
    for i in range(m):
        print(f"Prediction: {predict(X_train[i], w, b)}, Target: {y_train[i]}")

if __name__ == "__main__":
    main(dataset)
