from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import SparsePCA

# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load MNIST digits
    digits = load_digits()

    # Show some random digits
    selection = np.random.randint(0, 1797, size=100)

    fig, ax = plt.subplots(10, 10, figsize=(10, 10))

    samples = [digits.data[x].reshape((8, 8)) for x in selection]

    for i in range(10):
        for j in range(10):
            ax[i, j].set_axis_off()
            ax[i, j].imshow(samples[(i * 8) + j], cmap='gray')

    plt.show()

    # Perform a PCA on the digits dataset
    spca = SparsePCA(n_components=60, alpha=0.1)
    X_spca = spca.fit_transform(digits.data / 255)

    print('SPCA components shape:')
    print(spca.components_.shape)


