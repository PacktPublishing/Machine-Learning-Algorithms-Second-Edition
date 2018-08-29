from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.utils import resample


# For reproducibility
np.random.seed(1000)


nb_samples = 1000
weights = (0.95, 0.05)


if __name__ == '__main__':
    # Create an unbalanced dataset
    X, Y = make_classification(n_samples=nb_samples, n_features=2, n_redundant=0, weights=weights, random_state=1000)

    # Show the shapes
    print(X[Y == 0].shape)
    print(X[Y == 1].shape)

    # Show the dataset
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], label='Class 1')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], label='Class 2')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_title('Unbalanced dataset')
    ax.legend()
    ax.grid()

    plt.show()

    # Resample the dataset
    X_1_resampled = resample(X[Y == 1], n_samples=X[Y == 0].shape[0], random_state=1000)

    Xu = np.concatenate((X[Y == 0], X_1_resampled))
    Yu = np.concatenate((Y[Y == 0], np.ones(shape=(X[Y == 0].shape[0],), dtype=np.int32)))

    # Show the new shapes
    print(Xu[Yu == 0].shape)
    print(Xu[Yu == 1].shape)

