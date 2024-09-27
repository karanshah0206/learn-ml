import time
import numpy as np
import matplotlib.pyplot as plt
from multiple_linear_regression import gradient_descent
from normal_equation import normal_equation, compute_cost

def load_txt_dataset(txt_data_filename: str):
    """
    Loads a dataset from a txt file where the first line is a header with
    column names and following lines are numeric data.

    Args:
        txt_data_filename (str): Path to .txt data file

    Returns:
        tuple (ndarray, ndarray, list[str], str):
            - features (ndarray)
            - labels (ndarray)
            - feature_names (list[str])
            - target_name (str)
    """
    with open(txt_data_filename) as datafile:
        headers = datafile.readline().strip().split(",")
    dataset = np.loadtxt(txt_data_filename, skiprows=1, delimiter=",")
    return dataset[:,:-1], dataset[:,-1], headers[:-1], headers[-1]

def z_score_normalise(X: np.ndarray):
    mu = np.mean(X, axis=0) # find mean for each feature
    sigma = np.std(X, axis=0) # find std for each feature
    z_normalised = (X - mu) / sigma # z-score normalise all elements
    return z_normalised

def mean_score_normalise(X: np.ndarray):
    mu = np.mean(X, axis=0) # find mean for each feature
    r = np.max(X, axis=0) - np.min(X, axis=0) # find range of each feature
    mean_normalised = (X - mu) / r # mean normalise all elements
    return mean_normalised

def visualise_features_against_target(X: np.ndarray, y: np.ndarray, feature_names: list[str], target_name: str, plot_title: str):
    _, ax = plt.subplots(1, len(feature_names), figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X[:,i], y)
        ax[i].set_xlabel(feature_names[i])
    ax[0].set_ylabel(target_name)
    plt.suptitle(plot_title)
    plt.tight_layout()
    plt.show()

def main(txt_data_filename: str):
    X_train, y_train, features, target_column = load_txt_dataset(txt_data_filename)
    X_train_z_normalised = z_score_normalise(X_train)
    X_train_mean_normalised = mean_score_normalise(X_train)
    n = X_train.shape[1]

    visualise_features_against_target(X_train, y_train, features, target_column, "Unsacled Features against Target")
    visualise_features_against_target(X_train_z_normalised, y_train, features, target_column, "Z-score Normalised Features against Target")
    visualise_features_against_target(X_train_mean_normalised, y_train, features, target_column, "Mean Normalised Features against Target")

    print("Results by Normal Equation:")
    start_time = time.time()
    w, b = normal_equation(X_train, y_train)
    stop_time = time.time()
    print(f"Cost: {compute_cost(X_train, y_train, w, b)}, Time: {1000*(stop_time - start_time):.4f}ms")

    print("Results by Gradient Descent with Z-Score Normalisation:")
    alpha = 0.943
    start_time = time.time()
    _, _, J_history = gradient_descent(X_train_z_normalised, y_train, np.zeros(n), 0, alpha)
    stop_time = time.time()
    print(f"Cost: {J_history[-1]}, Time: {1000*(stop_time-start_time):.4f}ms, Alpha: {alpha}")

    print("Results by Gradient Descent with Mean Normalisation:")
    alpha = 1
    start_time = time.time()
    _, _, J_history = gradient_descent(X_train_mean_normalised, y_train, np.zeros(n), 0, alpha)
    stop_time = time.time()
    print(f"Cost: {J_history[-1]}, Time: {1000*(stop_time-start_time):.4f}ms, Alpha: {alpha}")

    print("Results by Gradient Descent without Feature Scaling (WARNING: This is going to take ages):")
    alpha = 8.83e-7
    start_time = time.time()
    _, _, J_history = gradient_descent(X_train, y_train, np.zeros(n), 0, alpha)
    stop_time = time.time()
    print(f"Cost: {J_history[-1]}, Time: {1000*(stop_time-start_time):.4f}ms, Alpha: {alpha}")

if __name__ == "__main__":
    main("../data/houses.txt")
