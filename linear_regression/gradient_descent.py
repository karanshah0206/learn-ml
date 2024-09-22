import math
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-10 # tolerance for imprecision in floating-point calculations

x_train = np.array([1.0, 2.0]) # size of house (1000 sq. ft.)
y_train = np.array([300.0, 500.0]) # price of house (1000 USD)

def compute_model_output(x: np.ndarray, w: float, b: float):
    """
    Computes the prediction of a linear regression model in one variable.

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

def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float):
    """
    Computes the squared error cost function for linear regression model with specified parameters.

    Args:
        x, y (ndarray (m,)): Feature and Targets, m samples
        w, b (scalar): Model parameters

    Returns:
        total_cost (float): Squared error cost for linear regression
    """
    m = x.shape[0]
    f_wb = compute_model_output(x, w, b)

    cost_sum = 0
    for i in range(m):
        cost_squared = (f_wb[i] - y[i]) ** 2
        cost_sum += cost_squared

    total_cost = cost_sum / (2 * m)
    return total_cost

def compute_gradient(x: np.ndarray, y: np.ndarray, w: float, b: float):
    """
    Computes the cost function gradient for univariate linear regression.

    Args:
        x, y (ndarray (m,)): Feature and Targets, m samples
        w, b (scalar): Model parameters

    Returns:
        dj_dw (scalar): Cost function gradient with respect to parameter w
        dj_db (scalar): Cost function gradient with respect to parameter b
    """
    m = x.shape[0]
    f_wb = compute_model_output(x, w, b)

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        dj_dw += (f_wb[i] - y[i]) * x[i]
        dj_db += f_wb[i] - y[i]

    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(x: np.ndarray, y: np.ndarray, w_in: float, b_in: float, alpha: float):
    """
    Performs gradient descent to optimise parameters w and b for univariate linear regression.

    Args:
        x, y (ndarray (m,)): Feature and Targets, m samples
        w_in, b_in (scalar): Initial values for model parameters
        alpha (float): Learning rate

    Returns:
        w (scalar): Updated value for parameter w
        b (scalar): Updated value for parameter b
        J_history (list[float]): History of cost vaules
        p_history (list): History of parameter values w[w, b]
    """
    i = 0
    w = w_in
    b = b_in
    J_history = list[float]()
    p_history = list[tuple[float, float]]()

    while True:
        old_w = w
        old_b = b
        i += 1

        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        J_history.append(compute_cost(x, y, w, b))
        p_history.append((w, b))

        if i % 1000 == 0:
            print(f"Iteration {i}: Cost {compute_cost(x, y, w, b): 0.2e} dj_dw: {dj_dw: 0.3e} dj_db: {dj_db: 0.3e} w: {w: 0.3e} b: {b: 0.5e}")

        if abs(old_w - w) <= EPSILON and abs(old_b - b) <= EPSILON:
            break

    return w, b, J_history, p_history

def visualise_gradient_descent(J_history: list[float]):
    """
    Visualises the change in cost over iterations during the gradient descent process.

    Args:
        J_history (list[float]): History of cost values recorded during gradient descent.
    """
    _, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))

    ax1.plot(J_history[:100])
    ax1.set_title("Cost vs Iteration Step (Start)")
    ax1.set_ylabel("Cost")
    ax1.set_xlabel("Iteration Step")

    ax2.plot(1000 + np.arange(len(J_history[1000:])), J_history[1000:])
    ax2.set_title("Cost vs Iteration Step (End)")
    ax2.set_ylabel("Cost")
    ax2.set_xlabel("Iteration Step")

    plt.show()

def inbounds(a: tuple[float, float], b: tuple[float, float], x_lim: tuple[float, float], y_lim: tuple[float, float]):
    """
    Checks if the points a and b are within the specified bounds.
    Helper function for contour visualiser.

    Args:
        a, b (tuple[float, float]): 2D coordinates
        x_lim, y_lim (tuple[float, float]): Upper and lower axis limits

    Returns:
        bool: True if both points are within the specified bounds, False otherwise
    """
    x_low, x_high = x_lim
    y_low, y_high = y_lim
    ax, ay = a
    bx, by = b
    return ax > x_low and ax < x_high and bx > x_low and bx < x_high and ay > y_low and ay < y_high and by > y_low and by < y_high

def visualise_cost_change_on_contour(x: np.ndarray, y: np.ndarray, p_history: list[tuple[float, float]]):
    """
    Visualises changes to cost function as a result of weights modified during the gradient descent process using contour plot.

    Args:
        x, y (ndarray (m,)): Feature and Targets, m samples
        p_history (list[tuple[float, float]]): History of parameter values (w, b) recorded during gradient descent
    """
    w_history = [p[0] for p in p_history]
    w_range = [math.floor(min(w_history)) - 20, math.ceil(max(w_history)) + 20, 0.5]
    w_final = w_history[-1]

    b_history = [p[1] for p in p_history]
    b_range = [math.floor(min(b_history)) - 20, math.ceil(max(b_history)) + 20, 0.5]
    b_final = b_history[-1]

    contours = [1, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    resolution = 0.5
    step = 10

    _, ax = plt.subplots(1, 1, figsize=(12, 4))

    b0, w0 = np.meshgrid(np.arange(*b_range), np.arange(*w_range))
    z = np.zeros_like(b0)
    for i in range(w0.shape[0]):
        for j in range(w0.shape[1]):
            z[i][j] = compute_cost(x, y, w0[i][j], b0[i][j])

    CS = ax.contour(w0, b0, z, contours, linewidths=2)
    ax.clabel(CS, inline=1, fmt="%1.0f", fontsize=10)
    ax.set_title("Cost $J(w, b)$ vs Weights $b$, $w$ with Path of Gradient Descent")
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")
    ax.hlines(b_final, ax.get_xlim()[0], w_final, lw=2, ls="dotted")
    ax.vlines(w_final, ax.get_ylim()[0], b_final, lw=2, ls="dotted")

    base = p_history[0]
    for point in p_history[0::step]:
        edist = np.sqrt((base[0] - point[0]) ** 2 + (base[1] - point[1]) ** 2)
        if edist > resolution or point == p_history[-1]:
            if inbounds(point, base, ax.get_xlim(), ax.get_ylim()):
                plt.annotate("", xy=point, xytext=base, xycoords="data", arrowprops={"arrowstyle": "->", "color": "r", "lw": 3}, va="center", ha="center")
            base = point

    plt.show()

w_final, b_final, J_history, p_history = gradient_descent(x_train, y_train, 0, 0, 1.0e-2)

print(f"(w, b) found by gradient decsent: ({w_final}, {b_final})")
print(f"Cost: {compute_cost(x_train, y_train, w_final, b_final)}")

visualise_gradient_descent(J_history)
visualise_cost_change_on_contour(x_train, y_train, p_history)

print(f"1000 sqft House Price Prediction: {w_final * 1.0 + b_final: .2f} Thousand Dollars")
print(f"1200 sqft House Price Prediction: {w_final * 1.2 + b_final: .2f} Thousand Dollars")
print(f"2000 sqft House Price Prediction: {w_final * 2.0 + b_final: .2f} Thousand Dollars")
