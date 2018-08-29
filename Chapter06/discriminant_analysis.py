from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score


# For reproducibility
np.random.seed(1000)


# Total number of samples
nb_samples = 1000


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=2, cluster_std=[1.0, 10.0], random_state=1000)

    # Show the dataset
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], label='Class 0')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], label='Class 1')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()
    ax.legend()

    plt.show()

    # Show the covariance matrices
    print('Covariance matrix for class 0:')
    print(np.cov(X[Y == 0].T))

    print('\nCovariance matrix for class 1:')
    print(np.cov(X[Y == 1].T))

    # Show the CV scores
    lda = LinearDiscriminantAnalysis()
    print('\nLDA average CV accuracy: %.3f' % cross_val_score(lda, X, Y, cv=10).mean())

    qda = QuadraticDiscriminantAnalysis()
    print('QDA average CV accuracy: %.3f' % cross_val_score(qda, X, Y, cv=10).mean())

