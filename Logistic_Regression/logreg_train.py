import numpy as np
import pandas as pd
import sys
import math_utils as mu


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Cost function
# @ = matrix mult op | [matrix].T = matrix transpose
def cost_function(x, y, theta):
    m = mu.len_set(y)
    predictions = sigmoid(x @ theta)
    cost = -(1/m) * (y.T @ np.log(predictions) +
                     (1 - y).T @ np.log(1 - predictions))
    return cost


# Gradient descent
# @ = matrix mult op | [matrix].T = matrix transpose
def gradient_descent(x, y, theta, alpha, iterations):
    m = mu.len_set(y)
    for _ in range(iterations):
        predictions = sigmoid(x @ theta)
        gradient = (1/m) * (x.T @ (predictions - y))
        theta -= alpha * gradient
    return theta


# Preprocessing the dataset
def preprocess_data(filename):
    data_frame = pd.read_csv(filename)
    data_frame = data_frame.drop(columns=['Index', 'First Name',
                                          'Last Name', 'Birthday'])
    data_frame['Best Hand'] = data_frame['Best Hand'].map({'Right': 1,
                                                           'Left': 0})

    # Encode target labels
    # houses = {'Gryffindor': 0, 'Hufflepuff': 1,
    #           'Ravenclaw': 2, 'Slytherin': 3}
    # data_frame['Hogwarts House'] = data_frame['Hogwarts House'].map(houses)

    x = data_frame.drop(columns=['Hogwarts House']).values  # Features (scores)
    y = data_frame['Hogwarts House'].values  # Target (house)

    # Normalize each column
    norm_x = x.copy()
    for i in range(x.shape[1]):
        column = x[:, i]
        mean = mu.mean(column)
        std = mu.std(column, mean)
        norm_x[:, i] = (column - mean) / std if std != 0 else column

    return norm_x, y  # houses


# Train one-vs-all models
def train_one_vs_all(x, y, num_classes, alpha, iterations):
    m, n = x.shape
    weights = np.zeros((num_classes, n + 1))  # Add one for bias
    x = np.c_[np.ones((m, 1)), x]  # Add bias term

    for c in range(num_classes):
        y_binary = (y == c).astype(int)  # Create binary labels for the class
        theta = np.zeros(n + 1)  # Initialize weights
        weights[c] = gradient_descent(x, y_binary, theta, alpha, iterations)
    return weights


# Main function
def main():
    try:
        if (len(sys.argv) != 2):
            raise AssertionError("wrong number of args")
        print("Preprocessing data...")
        x, y = preprocess_data(sys.argv[1])
        print("Training...")
        weights = train_one_vs_all(x, y, num_classes=4,
                                   alpha=0.1, iterations=1000)
        print("Saving weights...")
        np.save('weights.npy', weights)
        print("Weights saved to weights.npy")
    except Exception as e:
        print(f"logreg_train.py: error: {e}")


if __name__ == "__main__":
    main()
