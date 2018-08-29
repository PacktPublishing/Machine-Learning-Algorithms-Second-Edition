from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score


# For reproducibility
np.random.seed(1000)

nb_classifications = 100


if __name__ == '__main__':
    # Load dataset
    digits = load_digits()

    # Collect accuracies
    rf_accuracy = []
    et_accuracy = []

    for i in range(1, nb_classifications):
        a = cross_val_score(RandomForestClassifier(n_estimators=i), digits.data, digits.target, scoring='accuracy',
                            cv=10).mean()
        rf_accuracy.append(a)

        b = cross_val_score(ExtraTreesClassifier(n_estimators=i), digits.data, digits.target, scoring='accuracy',
                            cv=10).mean()
        et_accuracy.append(b)

    # Show results
    plt.figure(figsize=(30, 25))
    plt.xlabel('Number of trees')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(rf_accuracy, color='blue', label='Random Forest')
    plt.plot(et_accuracy, color='red', label='Extra Random Forest')
    plt.legend(loc="lower right")
    plt.show()