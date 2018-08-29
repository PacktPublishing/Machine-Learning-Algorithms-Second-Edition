from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical


# For reproducibility
np.random.seed(1000)

nb_samples = 2000


if __name__ == '__main__':
    # Create the dataset
    X = np.zeros(shape=(nb_samples, 2), dtype=np.float32)
    Y = np.zeros(shape=(nb_samples,), dtype=np.float32)

    t = 15.0 * np.random.uniform(0.0, 1.0, size=(int(nb_samples / 2), 1))

    X[0:int(nb_samples / 2), :] = t * np.hstack([-np.cos(t), np.sin(t)]) + \
                                  np.random.uniform(0.0, 1.8, size=(int(nb_samples / 2), 2))
    Y[0:int(nb_samples / 2)] = 0

    X[int(nb_samples / 2):, :] = t * np.hstack([np.cos(t), -np.sin(t)]) + \
                                 np.random.uniform(0.0, 1.8, size=(int(nb_samples / 2), 2))
    Y[int(nb_samples / 2):] = 1

    ss = StandardScaler()
    X = ss.fit_transform(X)

    X, Y = shuffle(X, Y, random_state=1000)

    # Show the dataset
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], label='Class 0')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], label='Class 1')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.legend()
    ax.grid()

    plt.show()

    # Perform a Logistic Regression cross-validation
    lr = LogisticRegression(penalty='l2', C=0.01, random_state=1000)
    print(np.mean(cross_val_score(lr, X, Y, cv=10)))

    # Show the classification result
    lr.fit(X, Y)
    Y_pred_lr = lr.predict(X)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(X[Y_pred_lr == 0, 0], X[Y_pred_lr == 0, 1], label='Class 0')
    ax.scatter(X[Y_pred_lr == 1, 0], X[Y_pred_lr == 1, 1], label='Class 1')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.legend()
    ax.grid()

    plt.show()

    # Create a Keras model
    model = Sequential()

    model.add(Dense(64, input_dim=2))
    model.add(Activation('relu'))

    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(16))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Split the dataset into train and test sets
    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, to_categorical(Y), test_size=0.2, random_state=1000)

    # Train the model
    model.fit(X_train, Y_train,
              epochs=100,
              batch_size=32,
              validation_data=(X_test, Y_test))

    # Show the classification result
    Y_pred_mlp = np.argmax(model.predict(X), axis=1)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(X[Y_pred_mlp == 0, 0], X[Y_pred_mlp == 0, 1], label='Class 0')
    ax.scatter(X[Y_pred_mlp == 1, 0], X[Y_pred_mlp == 1, 1], label='Class 1')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.legend()
    ax.grid()

    plt.show()

    # Show the decision surfaces
    Xm = np.linspace(-2.0, 2.0, 1000)
    Ym = np.linspace(-2.0, 2.0, 1000)
    Xmg, Ymg = np.meshgrid(Xm, Ym)
    X_eval = np.vstack([Xmg.ravel(), Ymg.ravel()]).T

    Y_eval_lr = lr.predict(X_eval)
    Y_eval_mlp = np.argmax(model.predict(X_eval), axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].scatter(X_eval[Y_eval_lr == 0, 0], X_eval[Y_eval_lr == 0, 1])
    ax[0].scatter(X_eval[Y_eval_lr == 1, 0], X_eval[Y_eval_lr == 1, 1])
    ax[0].set_xlabel(r'$x_0$')
    ax[0].set_ylabel(r'$x_1$')
    ax[0].set_title('Logistic Regression')

    ax[1].scatter(X_eval[Y_eval_mlp == 0, 0], X_eval[Y_eval_mlp == 0, 1])
    ax[1].scatter(X_eval[Y_eval_mlp == 1, 0], X_eval[Y_eval_mlp == 1, 1])
    ax[1].set_xlabel(r'$x_0$')
    ax[1].set_ylabel(r'$x_1$')
    ax[1].set_title('MLP')

    plt.show()

