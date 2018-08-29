from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score


# Set random seed for reproducibility
np.random.seed(1000)


# Download the dataset from: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
# Change <DATA_HOME> with the folder where the file is stored
file_path = '<DATA_HOME>/Concrete_Data.xls'
graphviz_path = '<DATA_HOME>/Concrete_Data.dot'


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_excel(file_path, header=0)
    X = df.iloc[:, 0:8].values
    Y = df.iloc[:, 8].values

    # Print the statistic summary
    print(df.describe())

    # Print the CV scores
    print(cross_val_score(DecisionTreeRegressor(criterion='mse', max_depth=11, random_state=1000), X, Y, cv=20))

    # Create train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=200, random_state=1000)

    # Train the Decision Tree Regressor
    dtr = DecisionTreeRegressor(criterion='mse', max_depth=11, random_state=1000)
    dtr.fit(X_train, Y_train)

    # Export the tree in Graphviz format
    # You can use http://www.webgraphviz.com to visualize the tree
    export_graphviz(dtr, out_file=graphviz_path,
                    feature_names=['Cement', 'Blast furnace slag', 'Fly ash', 'Water',
                                   'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age'])

    print('Training R^2 score: %.3f' % dtr.score(X_train, Y_train))
    print('Validation R^2 score: %.3f' % dtr.score(X_test, Y_test))

    # Compute the predictions
    Y_pred = dtr.predict(X)

    # Show the dataset, predictions and absolute errors
    fig, ax = plt.subplots(3, 1, figsize=(18, 15))

    ax[0].plot(Y)
    ax[0].set_title('Original dataset')
    ax[0].set_ylabel('Concrete Compressive Strength')
    ax[0].grid()

    ax[1].plot(Y_pred)
    ax[1].set_title('Predictions')
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Concrete Compressive Strength')
    ax[1].grid()

    ax[2].plot(np.abs(Y_pred - Y))
    ax[2].set_yticks(np.arange(0.0, 81.0, 10.0))
    ax[2].set_xlabel('Sample')
    ax[2].set_ylabel('Absolute error')
    ax[2].grid()

    plt.show()

    # Show the absolute error histogram
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.hist(np.abs(Y_pred - Y), bins='auto', log=True)
    ax.set_xlabel('Absolute error')
    ax.set_ylabel('Sample count')
    ax.grid()

    plt.show()