from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture


# Set random seed for reproducibility
np.random.seed(1000)


# Total number of samples
nb_samples = 800


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=3, cluster_std=2.2, random_state=1000)

    # Show the original dataset
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], c='r', s=20, marker='p', label='Class 0')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], c='g', s=20, marker='d', label='Class 1')
    ax.scatter(X[Y == 2, 0], X[Y == 2, 1], c='b', s=20, marker='s', label='Class 2')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.legend()
    ax.grid()

    plt.show()

    # Create a fit a Gaussian Mixture model
    gm = GaussianMixture(n_components=3, max_iter=1000, random_state=1000)
    gm.fit(X)

    # Print means, covariances, and weights
    print('Means:\n')
    print(gm.means_)

    print('\nCovariances:\n')
    print(gm.covariances_)

    print('\nWeights:\n')
    print(gm.weights_)

    # Show the clustered dataset with the final Gaussian distributions
    fig, ax = plt.subplots(figsize=(15, 8))

    c = gm.covariances_
    m = gm.means_

    g1 = Ellipse(xy=m[0], width=4 * np.sqrt(c[0][0, 0]), height=4 * np.sqrt(c[0][1, 1]), fill=False, linestyle='dashed',
                 linewidth=2)
    g1_1 = Ellipse(xy=m[0], width=3 * np.sqrt(c[0][0, 0]), height=3 * np.sqrt(c[0][1, 1]), fill=False,
                   linestyle='dashed', linewidth=3)
    g1_2 = Ellipse(xy=m[0], width=1.5 * np.sqrt(c[0][0, 0]), height=1.5 * np.sqrt(c[0][1, 1]), fill=False,
                   linestyle='dashed', linewidth=4)

    g2 = Ellipse(xy=m[1], width=4 * np.sqrt(c[1][0, 0]), height=4 * np.sqrt(c[1][1, 1]), fill=False, linestyle='dashed',
                 linewidth=2)
    g2_1 = Ellipse(xy=m[1], width=3 * np.sqrt(c[1][0, 0]), height=3 * np.sqrt(c[1][1, 1]), fill=False,
                   linestyle='dashed', linewidth=3)
    g2_2 = Ellipse(xy=m[1], width=1.5 * np.sqrt(c[1][0, 0]), height=1.5 * np.sqrt(c[1][1, 1]), fill=False,
                   linestyle='dashed', linewidth=4)

    g3 = Ellipse(xy=m[2], width=4 * np.sqrt(c[2][0, 0]), height=4 * np.sqrt(c[2][1, 1]), fill=False, linestyle='dashed',
                 linewidth=2)
    g3_1 = Ellipse(xy=m[2], width=3 * np.sqrt(c[2][0, 0]), height=3 * np.sqrt(c[2][1, 1]), fill=False,
                   linestyle='dashed', linewidth=3)
    g3_2 = Ellipse(xy=m[2], width=1.5 * np.sqrt(c[2][0, 0]), height=1.5 * np.sqrt(c[2][1, 1]), fill=False,
                   linestyle='dashed', linewidth=4)

    ax.add_artist(g1)
    ax.add_artist(g1_1)
    ax.add_artist(g1_2)
    ax.add_artist(g2)
    ax.add_artist(g2_1)
    ax.add_artist(g2_2)
    ax.add_artist(g3)
    ax.add_artist(g3_1)
    ax.add_artist(g3_2)

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], c='r', s=20, marker='p', label='Class 0')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], c='g', s=20, marker='d', label='Class 1')
    ax.scatter(X[Y == 2, 0], X[Y == 2, 1], c='b', s=20, marker='s', label='Class 2')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.legend()
    ax.grid()

    plt.show()

    # Compute AICs and BICs
    nb_components = [2, 3, 4, 5, 6, 7, 8]

    aics = []
    bics = []

    for n in nb_components:
        gm = GaussianMixture(n_components=n, max_iter=1000, random_state=1000)
        gm.fit(X)
        aics.append(gm.aic(X))
        bics.append(gm.bic(X))

    fig, ax = plt.subplots(2, 1, figsize=(15, 8))

    ax[0].plot(nb_components, aics)
    ax[0].set_ylabel('AIC')
    ax[0].grid()

    ax[1].plot(nb_components, bics)
    ax[1].set_xlabel('Number of components')
    ax[1].set_ylabel('BIC')
    ax[1].grid()

    plt.show()