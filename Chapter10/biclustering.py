from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster.bicluster import SpectralBiclustering


# Set random seed for reproducibility
np.random.seed(1000)


nb_users = 100
nb_products = 150
max_rating = 10


if __name__ == '__main__':
    # Create the user-product matrix
    up_matrix = np.random.randint(0, max_rating + 1, size=(nb_users, nb_products))
    mask_matrix = np.random.randint(0, 2, size=(nb_users, nb_products))
    up_matrix *= mask_matrix

    # Show the matrix
    fig, ax = plt.subplots(figsize=(12, 6))

    matx = ax.matshow(up_matrix)
    fig.colorbar(matx)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Products')
    ax.set_ylabel('Users')

    plt.show()

    # Perform a Spectral Biclustering
    sbc = SpectralBiclustering(n_clusters=10, random_state=1000)
    sbc.fit(up_matrix)

    # Show the clustered matrix
    up_clustered = np.outer(np.sort(sbc.row_labels_) + 1, np.sort(sbc.column_labels_) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    matx = ax.matshow(up_clustered)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Products')
    ax.set_ylabel('Users')

    plt.show()

    # Show some examples of users and products associated with ranking 6
    print(np.where(sbc.rows_[6, :] == True))
    print(np.where(sbc.columns_[6, :] == True))