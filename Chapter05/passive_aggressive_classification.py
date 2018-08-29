from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load and scale the dataset
    iris = load_iris()

    ss = StandardScaler()

    X = ss.fit_transform(iris['data'])
    Y = iris['target']

    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1000)

    # Create the model
    pac = PassiveAggressiveClassifier(C=0.05, loss='squared_hinge', max_iter=2000, random_state=1000)

    # Train with the start-up samples
    nb_initial_samples = int(X_train.shape[0] / 1.5)
    pac.fit(X_train[0:nb_initial_samples], Y_train[0:nb_initial_samples])

    # Continue with the incremental samples
    validation_accuracies = []

    for (x, y) in zip(X_train[nb_initial_samples:], Y_train[nb_initial_samples:]):
        pac.partial_fit(x.reshape(1, -1), y.ravel(), classes=np.unique(iris['target']))
        validation_accuracies.append(pac.score(X_test, Y_test))

    # Show the validation plot
    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(validation_accuracies)
    ax.set_xlabel('Online sample')
    ax.set_ylabel('Validation accuracy')
    ax.grid()

    plt.show()
