from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

# Install Imbalanced-Learning with: pip install -U imbalanced-learn
# For further information: http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html
from imblearn.over_sampling import SMOTE

from sklearn.datasets import make_classification


# For reproducibility
np.random.seed(1000)


nb_samples = 1000
weights = (0.95, 0.05)


if __name__ == '__main__':
    # Create an unbalanced dataset
    X, Y = make_classification(n_samples=nb_samples, n_features=2, n_redundant=0, weights=weights, random_state=1000)

    # Create and train a SMOTE instance
    smote = SMOTE()
    X_resampled, Y_resampled = smote.fit_sample(X, Y)

    # Show original and resampled datasets
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    ax[0].scatter(X[Y == 0, 0], X[Y == 0, 1], label='Class 1')
    ax[0].scatter(X[Y == 1, 0], X[Y == 1, 1], label='Class 2')
    ax[0].set_xlabel(r'$x_0$')
    ax[0].set_ylabel(r'$x_1$')
    ax[0].set_title('Unbalanced dataset')
    ax[0].legend()
    ax[0].grid()

    ax[1].scatter(X_resampled[Y_resampled == 0, 0], X_resampled[Y_resampled == 0, 1], label='Class 1')
    ax[1].scatter(X_resampled[Y_resampled == 1, 0], X_resampled[Y_resampled == 1, 1], label='Class 2')
    ax[1].set_xlabel(r'$x_0$')
    ax[1].set_ylabel(r'$x_1$')
    ax[1].set_title('SMOTE balancing')
    ax[1].legend()
    ax[1].grid()

    plt.show()



