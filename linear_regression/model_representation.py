import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0]) # size of house (1000 sq. ft.)
y_train = np.array([300.0, 500.0]) # price of house (1000 USD)

m = x_train.shape[0] # size of training dataset
w = 200 # gradient
b = 100 # y-intercept

# show variable values on stdout
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
print(f"m = {m}")

# visualise training data
plt.scatter(x_train, y_train, marker="x", c="r")
plt.title("Housing Prices by Size")
plt.ylabel("Price (in 1000 USD)")
plt.xlabel("Size (in 1000 sq. ft.)")
plt.show()

print(f"w = {w}")
print(f"b = {b}")

def compute_model_output(x: np.ndarray, w: float, b: float):
    """
    Computes the prediction of a linear regression model in one variable

    Args:
        x (ndarray (m,)): Data, m samples
        w, b (scalar): Model parameters

    Returns:
        f_wb (ndarray (m,)): Model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

# use linear regression model to predict output within training data domain
f_wb = compute_model_output(x_train, w, b)

# visualise computed model output
plt.plot(x_train, f_wb, c="b", label="Prediction Values")
plt.scatter(x_train, y_train, marker="x", c="r", label="Actual Values")
plt.title("Housing Prices by Size")
plt.ylabel("Price (in 1000 USD)")
plt.xlabel("Size (in 1000 sq. ft.)")
plt.legend()
plt.show()

# predicting house price for 1200 sqft
x_i = 1.2
prediction = w * x_i + b
print(f"Predicted price for house of 1200 sq. ft. = ${prediction * 1e3}")
