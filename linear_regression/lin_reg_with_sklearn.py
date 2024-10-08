import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from feature_scaling import load_txt_dataset

X_train, y_train, features, target_column = load_txt_dataset("../data/houses.txt")

# Feature scaling
scaler = StandardScaler()
X_train_normalised = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in raw: {np.ptp(X_train, axis=0)}")
print(f"Peak to Peak range by column in normalised: {np.ptp(X_train_normalised, axis=0)}")

# Training weights for linear regression model
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_train_normalised, y_train)
print("Gradient Descent parameters: ", sgdr.get_params())
print(f"Number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

# Trained model results
w_normalised = sgdr.coef_
b_normalised = sgdr.intercept_[0]
print(f"w: {w_normalised}, b: {b_normalised}")
predictions = sgdr.predict(X_train_normalised)
print(f"First 5 predictions: {predictions[:5]}")
print(f"First 5 targets: {y_train[:5]}")

# Visualise results
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i], y_train, label="target")
    ax[i].scatter(X_train[:,i], predictions, color="#ffa500", label="prediction")
    ax[i].set_xlabel(features[i])
ax[0].set_ylabel(target_column)
ax[0].legend()
fig.suptitle("Target vs Predictions using Z-Score Normalised Model")
plt.tight_layout()
plt.show()
