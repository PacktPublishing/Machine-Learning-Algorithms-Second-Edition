from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, HuberRegressor


# For reproducibility
np.random.seed(1000)

nb_samples = 500
nb_noise_samples = 50


def show_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.scatter(X, Y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid()

    plt.show()


if __name__ == '__main__':
    # Create dataset
    X = np.arange(-5, 5, 10.0 / float(nb_samples))

    Y = X + 2
    Y += np.random.uniform(-0.5, 0.5, size=nb_samples)

    noisy_samples = np.random.choice(np.arange(0, nb_samples), size=nb_noise_samples, replace=False)

    for i in noisy_samples:
        Y[i] += np.random.uniform(0, 10.0)

    # Show the dataset
    show_dataset(X, Y)

    # Create a linear regressor
    lr = LinearRegression(normalize=True)
    lr.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
    print('Standard regressor: y = %.3fx + %.3f' % (lr.coef_, lr.intercept_))

    # Create a Huber regressor
    hr = HuberRegressor(epsilon=1.25)
    hr.fit(X.reshape(-1, 1), Y)
    print('Huber regressor: y = %.3fx + %.3f' % (hr.coef_, hr.intercept_))