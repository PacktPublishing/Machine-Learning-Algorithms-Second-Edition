from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import Adam

# Install with pip -U install datapackage. For further information: https://datahub.io/core/global-temp#python
from datapackage import Package

from sklearn.preprocessing import MinMaxScaler


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 1600
nb_test_samples = 200
sequence_length = 20


if __name__ == '__main__':
    # Load the dataset
    package = Package('https://datahub.io/core/global-temp/datapackage.json')

    for resource in package.resources:
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            data = resource.read()

    # Extract the time series
    data_gcag = data[0:len(data):2][::-1]

    Y = np.zeros(shape=(len(data_gcag), 1), dtype=np.float32)

    for i, y in enumerate(data_gcag):
        Y[i - 1, 0] = y[2]

    # Scale between -1.0 and 1.0
    mmscaler = MinMaxScaler((-1.0, 1.0))
    Y = mmscaler.fit_transform(Y)

    # Show the time-series
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(Y)
    ax.grid()
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Monthly Avg Temperature Anomaly')

    plt.show()

    # Create the training and test sets
    X_ts = np.zeros(shape=(nb_samples - sequence_length, sequence_length, 1), dtype=np.float32)
    Y_ts = np.zeros(shape=(nb_samples - sequence_length, 1), dtype=np.float32)

    for i in range(0, nb_samples - sequence_length):
        X_ts[i] = Y[i:i + sequence_length]
        Y_ts[i] = Y[i + sequence_length]

    X_ts_train = X_ts[0:nb_samples - nb_test_samples, :]
    Y_ts_train = Y_ts[0:nb_samples - nb_test_samples]

    X_ts_test = X_ts[nb_samples - nb_test_samples:, :]
    Y_ts_test = Y_ts[nb_samples - nb_test_samples:]

    # Create the model
    model = Sequential()

    model.add(LSTM(8, stateful=True, batch_input_shape=(20, sequence_length, 1)))

    model.add(Dense(1))
    model.add(Activation('linear'))

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001, decay=0.0001),
                  loss='mse',
                  metrics=['mse'])

    # Train the model
    model.fit(X_ts_train, Y_ts_train,
              batch_size=20,
              epochs=100,
              shuffle=False,
              validation_data=(X_ts_test, Y_ts_test))

    # Show the predictions on the training set
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(Y_ts_train, label='True values')
    ax.plot(model.predict(X_ts_train, batch_size=20), label='Predicted values')
    ax.grid()
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Monthly Avg Temperature Anomaly')
    ax.legend()

    plt.show()

    # Show the predictions on the test set
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(Y_ts_test, label='True values')
    ax.plot(model.predict(X_ts_test, batch_size=20), label='Predicted values')
    ax.grid()
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Monthly Avg Temperature Anomaly')
    ax.legend()

    plt.show()

