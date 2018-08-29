from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering


# For reproducibility
np.random.seed(1000)

nb_samples = 1000


def show_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for i in range(nb_samples):
        if Y[i] == 0:
            ax.scatter(X[i, 0], X[i, 1], marker='o', color='r')
        else:
            ax.scatter(X[i, 0], X[i, 1], marker='^', color='b')

    plt.show()


def show_clustered_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='o', color='r')
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], marker='^', color='b')

    plt.show()


if __name__ == '__main__':
    warnings.simplefilter("ignore")

    # Create dataset
    X, Y = make_moons(n_samples=nb_samples, noise=0.05)

    # Show dataset
    show_dataset(X, Y)

    # Cluster the dataset for different gamma values
    Yss = []
    gammas = np.linspace(0, 12, 4)

    for gamma in gammas:
        sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=gamma)
        Yss.append(sc.fit_predict(X))

    # Show the result
    # The colors can be inverted with respect to the figure in the book
    fig, ax = plt.subplots(1, 4, figsize=(18, 8))

    for i in range(4):
        ax[i].scatter(X[Yss[i] == 1, 0], X[Yss[i] == 1, 1], marker='o', color='r')
        ax[i].scatter(X[Yss[i] == 0, 0], X[Yss[i] == 0, 1], marker='^', color='b')
        ax[i].grid()
        ax[i].set_xlabel('X')
        ax[i].set_ylabel('Y')
        ax[i].set_title('Gamma = {}'.format(i * 4))

    plt.show()

    # Create and train Spectral Clustering
    sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')
    Ys = sc.fit_predict(X)

    # Show clustered dataset
    show_clustered_dataset(X, Y)
