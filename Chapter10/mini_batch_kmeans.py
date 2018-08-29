from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 2000
batch_size = 80


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=5, cluster_std=1.5, random_state=1000)

    # Create an instance of Mini-Batch k-Means
    mbkm = MiniBatchKMeans(n_clusters=5, max_iter=1000, batch_size=batch_size, random_state=1000)

    # Train the model
    X_batch = []
    Y_preds = []

    for i in range(0, nb_samples, batch_size):
        mbkm.partial_fit(X[i:i + batch_size])

        X_batch.append(X[:i + batch_size])
        Y_preds.append(mbkm.predict(X[:i + batch_size]))

    # Show the training steps
    fig, ax = plt.subplots(5, 5, figsize=(20, 12))

    for i in range(5):
        for j in range(5):
            idx = (i * 5) + j

            for k in range(5):
                ax[i][j].scatter(X_batch[idx][Y_preds[idx] == k, 0], X_batch[idx][Y_preds[idx] == k, 1], s=3)

            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].set_title('{} samples'.format(batch_size * (idx + 1)))

    plt.show()

    # Compute the Adjusted-Rand score and compare it with a standard K-Means
    print(adjusted_rand_score(mbkm.predict(X), Y))

    km = KMeans(n_clusters=5, max_iter=1000, random_state=1000)
    km.fit(X)

    print(adjusted_rand_score(km.predict(X), Y))

