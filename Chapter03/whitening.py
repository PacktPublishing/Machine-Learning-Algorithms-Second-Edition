from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


# For reproducibility
np.random.seed(1000)


nb_samples = 1000


def zero_center(X):
    return X - np.mean(X, axis=0)


def whiten(X, correct=True):
    Xc = zero_center(X)
    _, L, V = np.linalg.svd(Xc)
    W = np.dot(V.T, np.diag(1.0 / L))
    return np.dot(Xc, W) * np.sqrt(X.shape[0]) if correct else 1.0


if __name__ == '__main__':
    # Create the dataset
    X = np.random.normal(0.0, [2.5, 1.0], size=(nb_samples, 2))

    theta = np.pi / 4.0
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    Xr = np.dot(X, R)

    # Create a whitened version
    Xw = whiten(Xr)

    # Print the whitened covariance matrix
    print(np.cov(Xw.T))

    # Show original and whitened datasets
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].scatter(Xr[:, 0], Xr[:, 1])
    ax[0].set_xticks(np.arange(-10, 10), 2)
    ax[0].set_yticks(np.arange(-8, 8), 2)
    ax[0].set_xlabel(r'$x_1$')
    ax[0].set_ylabel(r'$x_2$')
    ax[0].set_title(r'Original dataset')
    ax[0].grid()

    ax[1].scatter(Xw[:, 0], Xw[:, 1])
    ax[1].set_xticks(np.arange(-10, 10), 2)
    ax[1].set_yticks(np.arange(-8, 8), 2)
    ax[1].set_xlabel(r'$x_1$')
    ax[1].set_ylabel(r'$x_2$')
    ax[1].set_title(r'Whitened dataset')
    ax[1].grid()

    plt.show()

