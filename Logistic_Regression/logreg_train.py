import numpy as np
import pandas as pd
import sys
import os
import math_utils as mu


# Cost function
# @ = matrix mult op | [matrix].T = matrix transpose
def cost_function(x, y, theta):
    m = mu.len_set(y)
    predictions = mu.sigmoid(x @ theta)
    # predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
    cost = -(1/m) * (y.T @ np.log(predictions) +
                     (1 - y).T @ np.log(1 - predictions))
    return cost


# Gradient descent
# @ = matrix mult op | [matrix].T = matrix transpose
def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        predictions = mu.sigmoid(x @ theta)
        # print(f"aaaaaaa{predictions}")
        gradient = (1/m) * (x.T @ (predictions - y))
        # if np.any(np.isnan(gradient)):
        #     print(f"NaN detected in gradient at iteration {i}")
        theta -= alpha * gradient
        # if np.any(np.isnan(theta)):
        #     print(f"NaN detected in theta at iteration {i}")
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
    x = pd.DataFrame(x)
    for column in x.columns:
        column_data = x[column].values
        mean_val = mu.mean(column_data)
        std_val = mu.std(column_data, mean_val)
        x[column] = (column_data - mean_val) / std_val
    return x, y  # houses


# Train one-vs-all models
def train_ova(x, y, num_classes, alpha, iterations):
    m, n = x.shape
    weights = np.zeros((num_classes, n + 1))  # Add one for bias
    x = np.c_[np.ones((m, 1)), x]


    for c in range(num_classes):
        y_binary = (y == c).astype(int)  # Create binary labels for the class
        theta = np.zeros(x.shape[1])  # Initialize weights
        weights[c] = gradient_descent(x, y_binary, theta, alpha, iterations)
    return weights


def replace_nan_with_mean(x):
    for i in range(x.shape[1]):  # Iterate through each column
        col = x.iloc[:, i]  # Access column using `.iloc`
        if col.isnull().any():  # Check for NaN values
            mean_value = col.mean()  # Calculate mean of the column
            x.iloc[:, i] = col.fillna(mean_value)  # Replace NaNs with the mean
    return x


# Main function
def main():
    # try:
        if (len(sys.argv) != 2):
            raise AssertionError("wrong number of args")
        print("Preprocessing data...")
        x, y = preprocess_data(sys.argv[1])
        x = replace_nan_with_mean(x)
        print("Training...")
        weights = train_ova(x, y, num_classes=4,
                                   alpha=0.1, iterations=1000)
        print("Saving weights...")
        if os.path.exists('weights.npy'):
            os.remove('weights.npy')
        np.save('weights.npy', weights)
        print("Weights saved to weights.npy")
    # except Exception as e:
    #     print(f"logreg_train.py: error: {e}")


if __name__ == "__main__":
    main()
