from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    digits = load_digits()

    # Scale the dataset
    ss = StandardScaler(with_std=False)
    X = ss.fit_transform(digits['data'])

    # Create and train the model
    knn = NearestNeighbors(n_neighbors=25, leaf_size=30, algorithm='ball_tree')
    knn.fit(X)

    # Create a noisy sample (and show it)
    X_noise = X[50] + np.random.normal(0.0, 1.5, size=(64,))

    fig, ax = plt.subplots(1, 2, figsize=(4, 8))

    ax[0].imshow(digits['images'][50], cmap='gray')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(ss.inverse_transform(X_noise).reshape((8, 8)), cmap='gray')
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.show()

    # Compute the neighbors
    distances, neighbors = knn.kneighbors(X_noise.reshape(1, -1), return_distance=True)

    print('Distances:\n')
    print(distances[0])

    # Show the neighbors
    fig, ax = plt.subplots(5, 5, figsize=(8, 8))

    for y in range(5):
        for x in range(5):
            idx = neighbors[0][(x + (y * 5))]
            ax[y, x].matshow(digits['images'][idx], cmap='gray')
            ax[y, x].set_xticks([])
            ax[y, x].set_yticks([])

    plt.show()