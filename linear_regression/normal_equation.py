import numpy as np

# Initialise dummy house prices dataset
dataset = {
    "size_sqft": [2104., 1416., 852., 1534., 2209.],
    "bedrooms_count": [5., 3., 2., 3., 4.],
    "floors_count": [1., 2., 1., 2., 2.],
    "property_age_years": [45., 40., 35., 30., 50.],
    "price_1k_usd": [460., 232., 178., 351., 520.]
}

def predict(x: np.ndarray, w: np.ndarray, b: float):
    return np.dot(x, w) + b

def predict_batch(X: np.ndarray, w: np.ndarray, b: float):
    return np.array([predict(X[i], w, b) for i in range(X.shape[0])])

def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    return sum((prediction - y[i]) ** 2 for i, prediction in enumerate(predict_batch(X, w, b))) / (2 * len(y))

def normal_equation(X_train: np.ndarray, y_train: np.ndarray):
    X = np.concatenate([np.ones((len(X_train), 1)), X_train], axis=1)
    X_T = X.transpose()
    weights = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y_train)
    return weights[1:], weights[0]

def main(dataset: dict[str, list[float]]):
    target_column = "price_1k_usd"

    m = len(dataset[target_column]) # number of records

    X_train = np.array([[dataset[key][i] for key in dataset.keys() if key != target_column] for i in range(m)])
    y_train = np.array(dataset[target_column])

    # Optimise weights
    w, b = normal_equation(X_train, y_train)
    print("Final w weights:", w, "Final b weight:", b, "Cost with final weights:", compute_cost(X_train, y_train, w, b))

    # Predictions with finalised weights
    for i in range(m):
        print(f"Prediction: {predict(X_train[i], w, b)}, Target: {y_train[i]}")

if __name__ == "__main__":
    main(dataset)
