from __future__ import print_function

import numpy as np

from scipy.optimize import minimize


# For reproducibility
np.random.seed(1000)


nb_samples = 100

# Create the dataset
X_data = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=nb_samples)


def negative_log_likelihood(v):
    l = 0.0
    f1 = 1.0 / np.sqrt(2.0 * np.pi * v[1])
    f2 = 2.0 * v[1]

    for x in X_data:
        l += np.log(f1 * np.exp(-np.square(x - v[0]) / f2))

    return -l


if __name__ == '__main__':
    # Create the dataset
    res = minimize(fun=negative_log_likelihood, x0=np.array([0.0, 1.0]))

    print(res)

