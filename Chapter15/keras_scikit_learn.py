from __future__ import print_function

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier


# For reproducibility
np.random.seed(1000)

nb_samples = 2000


def build_model(lr=0.001):
    model = Sequential()

    model.add(Dense(64, input_dim=2))
    model.add(Activation('relu'))

    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(16))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


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

    # Wrap the Keras model
    skmodel = KerasClassifier(build_fn=build_model, epochs=100, batch_size=32, lr=0.001)

    # Perform a grid search
    parameters = {
        'lr': [0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128]
    }

    gs = GridSearchCV(skmodel, parameters, cv=5)
    gs.fit(X, to_categorical(Y, 2))

    # Show the best score and parameters
    print(gs.best_score_)
    print(gs.best_params_)

