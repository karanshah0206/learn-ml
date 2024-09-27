import math
import numpy as np
import matplotlib.pyplot as plt
from feature_scaling import z_score_normalise
from multiple_linear_regression import compute_cost, compute_gradient

def capped_gradient_descent(X: np.ndarray, y: np.ndarray, w_init: np.ndarray, b_init: int, alpha: float, iterations: int):
    w = w_init.copy()
    b = b_init

    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration {i}, Cost: {compute_cost(X, y, w, b):0.5e}")

    return w, b

def run_gradient_descent(X: np.ndarray, y: np.ndarray, iterations: int, alpha: float):
    n = X.shape[1]
    initial_w = np.zeros(n)
    initial_b = 0
    w_out, b_out = capped_gradient_descent(X, y, initial_w, initial_b, alpha, iterations)
    print(f"w: {w_out}, b: {b_out}")
    return w_out, b_out

def visualise_predictions(x: np.ndarray, y: np.ndarray, X: np.ndarray, model_w: np.ndarray, model_b: np.ndarray, title: str):
    plt.scatter(x, y, marker="x", c="r", label="Actual Value")
    plt.title(title)
    plt.plot(x, X@model_w + model_b, label="Predicted Value")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# Generate dummy dataset where dependent variable is linearly related to the square of covariant
x = np.arange(0, 20)
y = 1 + x**2

# First attempt: run linear regression without feature engineering
X = x.reshape(-1, 1)
model_w, model_b = run_gradient_descent(X, y, 1000, 1e-2)
# The graph shows that a linear model isn't the best fit for this data.
visualise_predictions(x, y, X, model_w, model_b, "No Feature Engineering")

# Second attempt: run polynomial regression of order 2
x_squared = x**2 # adding engineered feature x^2
X = x_squared.reshape(-1, 1) # notice we are not selecting the feature x itself
model_w, model_b = run_gradient_descent(X, y, 10000, 1e-5)
# The graph shows a polynomial model (of order 2) is a much better fit.
visualise_predictions(x, y, X, model_w, model_b, "Added x^2 Feature")

# Third attempt: run polynomial regression of order 3
x_cubed = x**3 # adding engineered feature x^3
X = np.c_[x, x_squared, x_cubed] # selecting features x, x^2, and x^2 (the latter two are engineered)
model_w, model_b = run_gradient_descent(X, y, 10000, 1e-7)
# Note that the w_1 term (i.e., coefficient to x^2) is much larger than other weights.
# Gradient descent is automatically emphasising on the important weight.
# This is a useful hint; we should probably stick to a second-order polynomial for this data.
visualise_predictions(x, y, X, model_w, model_b, "x, x^2, x^3 Features")

# Building intuition on which order polynomial is a good choice through visualisations.
X_features = ["x", "x^2", "x^3"]
fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i], y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
# We can see that x^2 is the closest linear match to the target variable.
# Polynomial regression is just a special case of linear regression, so the engineered higher-order
# features should have a linear relationship with respect to the target to create a good model.
plt.show()

# Exploring the importance of feature scaling with polynomial regression
X_normalised = z_score_normalise(X)
print(f"Peak to Peak range by column without scaling: {np.ptp(X, axis=0)}")
print(f"Peak to Peak range by column with scaling: {np.ptp(X_normalised, axis=0)}")

# Fourth attempt: run polynomial regression of order 3 with normalisation
# Notice the aggresive alpha (couldn't be used earlier due to unscaled features)
model_w, model_b = run_gradient_descent(X_normalised, y, 100000, 0.1)
visualise_predictions(x, y, X_normalised, model_w, model_b, "Normalised x, x^2, x^3 Features")

# Having some fun with polynomial regression & feature engineering
x = np.arange(0, 20)
y = np.cos(x / 2)
X = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X_normalised = z_score_normalise(X)
model_w, model_b = run_gradient_descent(X_normalised, y, 1000000, 0.1) # will take a moment
visualise_predictions(x, y, X_normalised, model_w, model_b, "Polynomial Regression with Engineered & Scaled Features - Go Grazy :)")
