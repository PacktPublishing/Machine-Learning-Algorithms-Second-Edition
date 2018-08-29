from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os

from shutil import copyfileobj
from six.moves import urllib

from sklearn.datasets.base import get_data_home
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import FastICA


# Set random seed for reproducibility
np.random.seed(1000)


# mldata.org can be subject to outages
# Alternative original MNIST source (provided by Aurélien Geron)
def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)


def zero_center(Xd):
    return Xd - np.mean(Xd, axis=0)


if __name__ == '__main__':
    # Load the dataset
    mnist = fetch_mnist()
    digits = fetch_mldata("MNIST original")
    X = zero_center(digits['data'].astype(np.float64))
    np.random.shuffle(X)

    # Peform Fast ICA with 64 components
    fastica = FastICA(n_components=256, max_iter=5000, random_state=1000)
    fastica.fit(X)

    # Plot the indipendent components
    fig, ax = plt.subplots(8, 8, figsize=(11, 11))

    for i in range(8):
        for j in range(8):
            ax[i, j].imshow(fastica.components_[(i * 8) + j].reshape((28, 28)), cmap='gray')
            ax[i, j].axis('off')

    plt.show()
