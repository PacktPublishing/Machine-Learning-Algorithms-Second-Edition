from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    digits = load_digits()
    X = digits['data'] / np.max(digits['data'])

    # Perform a t-SNE
    tsne = TSNE(n_components=2, perplexity=20, random_state=1000)
    X_tsne = tsne.fit_transform(X)

    # Plot the t-SNE result
    fig, ax = plt.subplots(figsize=(18, 10))

    for i in range(400):
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], color=cm.rainbow(digits['target'] * 10), marker='o', s=20)
        ax.annotate('%d' % digits['target'][i], xy=(X_tsne[i, 0] + 1, X_tsne[i, 1] + 1))

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()

    plt.show()