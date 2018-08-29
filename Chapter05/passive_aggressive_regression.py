from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import PassiveAggressiveRegressor


# For reproducibility
np.random.seed(1000)

nb_samples_1 = 300
nb_samples_2 = 500


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_regression(n_samples=nb_samples_1, n_features=5, random_state=1000)

    # Create the model
    par = PassiveAggressiveRegressor(C=0.01, loss='squared_epsilon_insensitive', epsilon=0.001, max_iter=2000,
                                     random_state=1000)

    # Fit the model incrementally and collect the squared errors
    squared_errors = []

    for (x, y) in zip(X, Y):
        par.partial_fit(x.reshape(1, -1), y.ravel())
        y_pred = par.predict(x.reshape(1, -1))
        squared_errors.append(np.power(y_pred - y, 2))

    # Show the error plot
    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(squared_errors)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Squared error')
    ax.grid()

    plt.show()

    # Repeat the example with a discontinuous dataset
    X1, Y1 = make_regression(n_samples=nb_samples_2, n_features=5, random_state=1000)
    X2, Y2 = make_regression(n_samples=nb_samples_2, n_features=5, random_state=1000)

    X2 += np.max(X1)
    Y2 += 0.5

    X = np.concatenate((X1, X2))
    Y = np.concatenate((Y1, Y2))

    par = PassiveAggressiveRegressor(C=0.01, loss='squared_epsilon_insensitive', epsilon=0.001, max_iter=2000,
                                     random_state=1000)

    # Fit the model incrementally and collect the squared errors
    squared_errors = []

    for (x, y) in zip(X, Y):
        par.partial_fit(x.reshape(1, -1), y.ravel())
        y_pred = par.predict(x.reshape(1, -1))
        squared_errors.append(np.power(y_pred - y, 2))

    # Show the error plot
    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(squared_errors)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Squared error')
    ax.grid()

    plt.show()