from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.datasets import make_circles
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score

# For reproducibility
np.random.seed(1000)


nb_samples = 3000


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_circles(n_samples=nb_samples, noise=0.05)

    # Show the dataset
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.scatter(X[:, 0], X[:, 1], marker='o', color='b')
    plt.show()

    # Unstructured clustering with average linkage
    print('Unstructured clustering with average linkage')
    ac = AgglomerativeClustering(n_clusters=20, linkage='average')
    Y_pred = ac.fit_predict(X)

    print('Silhouette score: %.3f' % silhouette_score(X, Y_pred))

    # Plot the clustered dataset
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.scatter(X[:, 0], X[:, 1], marker='o', cmap=cm.spectral, c=ac.labels_)
    plt.show()

    # Connectivity constraints
    print('Imposing connectivity constraints')

    acc = []
    k = [50, 100, 200, 500]

    ac = AgglomerativeClustering(n_clusters=20, connectivity=None, linkage='average')
    ac.fit(X)

    for i in k:
        kng = kneighbors_graph(X, i)
        ac1 = AgglomerativeClustering(n_clusters=20, connectivity=kng, linkage='average')
        Y_pred = ac1.fit_predict(X)
        print('Silhouette score (k=%d): %.3f' % (i, silhouette_score(X, Y_pred)))
        acc.append(ac1)

    # Show the four plots
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    ax[0, 0].grid()
    ax[0, 0].set_title('K = 50')
    ax[0, 0].set_xlabel('X')
    ax[0, 0].set_ylabel('Y')
    ax[0, 0].scatter(X[:, 0], X[:, 1], marker='o', cmap=cm.spectral, c=acc[0].labels_)

    ax[0, 1].grid()
    ax[0, 1].set_title('K = 100')
    ax[0, 1].set_xlabel('X')
    ax[0, 1].set_ylabel('Y')
    ax[0, 1].scatter(X[:, 0], X[:, 1], marker='o', cmap=cm.spectral, c=acc[1].labels_)

    ax[1, 0].grid()
    ax[1, 0].set_title('K = 200')
    ax[1, 0].set_xlabel('X')
    ax[1, 0].set_ylabel('Y')
    ax[1, 0].scatter(X[:, 0], X[:, 1], marker='o', cmap=cm.spectral, c=acc[2].labels_)

    ax[1, 1].grid()
    ax[1, 1].set_title('K = 500')
    ax[1, 1].set_xlabel('X')
    ax[1, 1].set_ylabel('Y')
    ax[1, 1].scatter(X[:, 0], X[:, 1], marker='o', cmap=cm.spectral, c=acc[3].labels_)
    plt.show()

