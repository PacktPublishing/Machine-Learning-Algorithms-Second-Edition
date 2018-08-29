from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    wine = load_wine()

    # Shuffle the dataset and compute the learning curves
    X, Y = shuffle(wine['data'], wine['target'])
    tsize, training_score, test_score = learning_curve(LogisticRegression(), X, Y, cv=20, random_state=1000)

    # Show the learning curve
    avg_tr_scores = np.mean(training_score, axis=1)
    avg_test_scores = np.mean(test_score, axis=1)

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(tsize, avg_tr_scores, label='Training score')
    ax.plot(tsize, avg_test_scores, label='CV score')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid()

    plt.show()