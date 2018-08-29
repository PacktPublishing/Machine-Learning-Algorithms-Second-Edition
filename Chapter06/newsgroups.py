from __future__ import print_function

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


# For reproducibility
np.random.seed(1000)


def plot_confusion_matrix(Y_test, Y_pred, targets):
    cmatrix = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
    cm_fig, cm_ax = plt.subplots(figsize=(12, 12))
    cm_ax.matshow(cmatrix, cmap=cm.GnBu)

    x = y = np.arange(0, len(targets))
    plt.xticks(x, targets, rotation='vertical')
    plt.yticks(y, targets)

    for i in range(len(targets)):
        for j in range(len(targets)):
            cm_ax.text(x=j, y=i, s=cmatrix[i, j], va='center', ha='center', size='x-large')

    plt.show()


if __name__ == '__main__':
    # Load the dataset
    train_data = fetch_20newsgroups_vectorized(subset='train')
    test_data = fetch_20newsgroups_vectorized(subset='test')

    # Create and train the model
    mnb = MultinomialNB(alpha=0.01)
    mnb.fit(train_data['data'], train_data['target'])

    print(mnb.score(test_data['data'], test_data['target']))

    # Plot the confusion matrix
    plot_confusion_matrix(test_data['target'], mnb.predict(test_data['data']), list(test_data['target_names']))