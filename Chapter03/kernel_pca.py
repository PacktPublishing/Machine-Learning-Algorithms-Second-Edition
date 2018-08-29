from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Create a dummy dataset
    Xb, Yb = Xb, Yb = make_circles(n_samples=500, factor=0.1, noise=0.05)

    # Show the dataset
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(Xb[:, 0], Xb[:, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid()

    plt.show()

    # Perform a kernel PCA (with radial basis function)
    kpca = KernelPCA(n_components=2, kernel='rbf', fit_inverse_transform=True, gamma=1.0)
    X_kpca = kpca.fit_transform(Xb)

    # Plot the dataset after PCA
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(kpca.X_transformed_fit_[:, 0], kpca.X_transformed_fit_[:, 1])
    ax.set_xlabel('First component')
    ax.set_ylabel('Second component')
    ax.grid()

    plt.show()