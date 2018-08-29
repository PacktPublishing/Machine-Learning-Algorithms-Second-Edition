from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# For reproducibility
np.random.seed(1000)

# Download the dataset from: https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
# Change <DATA_HOME> with the folder where the file is stored
file_path = '<DATA_HOME>/airfoil_self_noise.dat'


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv(file_path, sep='\t', header=None)

    # Show the statistics
    print(df.describe())

    # Extract the independent and dependent variables
    X = df.iloc[:, 0:5].values
    Y = df.iloc[:, 5].values

    # Scale the data
    ssx, ssy = StandardScaler(), StandardScaler()

    Xs = ssx.fit_transform(X)
    Ys = ssy.fit_transform(Y.reshape(-1, 1))

    # Create train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(Xs, Ys.ravel(), test_size=300, random_state=1000)

    # Instantiate and train the SVR
    svr = SVR(kernel='rbf', gamma=0.75, C=2.8, cache_size=500, epsilon=0.1)
    svr.fit(X_train, Y_train)

    # Print the R^2 scores
    print('Training R^2 score: %.3f' % svr.score(X_train, Y_train))
    print('Test R^2 score: %.3f' % svr.score(X_test, Y_test))

    # Show both original dataset and predictions
    fig, ax = plt.subplots(2, 1, figsize=(15, 9))

    ax[0].plot(ssy.inverse_transform(Ys))
    ax[0].set_title('Original dataset')
    ax[0].set_ylabel('Scaled sound pressure (dB)')
    ax[0].grid()

    ax[1].plot(ssy.inverse_transform(svr.predict(Xs)))
    ax[1].set_title('Predictions')
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Scaled sound pressure (dB)')
    ax[1].grid()

    plt.show()

    # Show the absolute errors
    fig, ax = plt.subplots(figsize=(15, 4))

    Y = np.squeeze(ssy.inverse_transform(Ys))
    Yp = ssy.inverse_transform(svr.predict(Xs))

    ax.plot(np.abs(Y - Yp))
    ax.set_title('Absolute errors')
    ax.set_xlabel('Sample')
    ax.set_ylabel(r'$|Y-Yp|$')
    ax.grid()

    plt.show()