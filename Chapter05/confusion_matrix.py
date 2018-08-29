from __future__ import print_function

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# For reproducibility
np.random.seed(1000)


def plot_confusion_matrix(Y_test, Y_pred, targets):
    cmatrix = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
    cm_fig, cm_ax = plt.subplots(figsize=(8.0, 8.0))
    cm_ax.matshow(cmatrix, cmap=cm.GnBu)

    cm_ax.set_xticklabels([''] + targets)
    cm_ax.set_yticklabels([''] + targets)

    for i in range(len(targets)):
        for j in range(len(targets)):
            cm_ax.text(x=j, y=i, s=cmatrix[i, j], va='center', ha='center', size='x-large')

    plt.title('Confusion matrix')
    plt.show()


if __name__ == '__main__':
    # Load the dataset
    wine = load_wine()

    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(wine['data'], wine['target'], test_size=0.25)

    # Train the model
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)

    # Plot the confusion matrix
    targets = list(wine['target_names'])
    plot_confusion_matrix(lr.predict(X_test), Y_test, targets)



