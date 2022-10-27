from __future__ import print_function

import numpy as np

from sklearn.impute import SimpleImputer

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    data = np.array([[1, np.nan, 2], [2, 3, np.nan], [-1, 4, 2]])
    print(data)

    # Imputer with mean-strategy
    print('Mean strategy')
    imp = SimpleImputer(strategy='mean')
    print(imp.fit_transform(data))

    # Imputer with median-strategy
    print('Median strategy')
    imp = SimpleImputer(strategy='median')
    print(imp.fit_transform(data))

    # Imputer with most-frequent-strategy
    print('Most-frequent strategy')
    imp = SimpleImputer(strategy='most_frequent')
    print(imp.fit_transform(data))

