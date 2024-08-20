import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2]) # size of house (1000 sq. ft.)
y_train = np.array([250, 300, 480,  430,   630, 730]) # price of house (1000 USD)

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

def visualise_cost(x: np.ndarray, y: np.ndarray, w_min: float, w_max: float, b_min: float, b_max: float):
    """
    Visualise effect of manipulating linear regression model parameters on cost function.

    Args:
        x, y: (ndarray (m,)): Feature and Targets, m samples
        w_min, b_min (scalar): Minimum value for model parameters
        w_max, b_max (scalar): Maximum value for model parameters
    """
    w = np.linspace(w_min, w_max, 100)
    b = np.linspace(b_min, b_max, 100)
    z = np.zeros((len(w), len(b)))
    W, B = np.meshgrid(w, b)

    for j in range(len(w)):
        for i in range(len(b)):
            z[i, j] = compute_cost(x, y, w[j], b[i])

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    ax1.plot_surface(W, B, z, cmap="Spectral_r", alpha=0.7, antialiased=False)
    ax1.plot_wireframe(W, B, z, color='k', alpha=0.1)
    ax1.set_xlabel("$w$")
    ax1.set_ylabel("$b$")
    ax1.set_zlabel("$J$")
    ax1.set_title("Cost 3D")

    ax2.contour(W, B, np.log(z), levels=12, cmap="Spectral_r")
    ax2.set_xlabel("$w$")
    ax2.set_ylabel("$b$")
    ax2.set_title("Cost Contour")

    plt.show()

visualise_cost(x_train, y_train, -100, 500, -200, 300)
