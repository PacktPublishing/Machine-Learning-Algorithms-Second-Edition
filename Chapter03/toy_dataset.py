from __future__ import print_function

import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    boston = load_boston()
    X = boston.data
    Y = boston.target

    print(X.shape)
    print(Y.shape)

    # Create train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1000)

    # Use a random state
    rs = check_random_state(1000)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=rs)