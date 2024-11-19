import numpy as np
import pandas as pd
import sys
import os
import math_utils as mu


# Cost function
# @ = matrix mult op | [matrix].T = matrix transpose
# def cost_function(x, y, theta):
#     """
#     Computes the cost function for logistic regression
#     using the sigmoid function.

#     Parameters:
#         x (numpy.ndarray): Input feature matrix (m x n).
#         y (numpy.ndarray): Actual target labels (m x 1).
#         theta (numpy.ndarray): Model parameters (n x 1).

#     Returns:
#         numpy.ndarray: The computed cost.
#     """
#     m = mu.len_set(y)
#     predictions = mu.sigmoid(x @ theta)
#     cost = -(1/m) * (y.T @ np.log(predictions) +
#                      (1 - y).T @ np.log(1 - predictions))
#     return cost


# Gradient descent
# @ = matrix mult op | [matrix].T = matrix transpose
def gradient_descent(x, y, theta, alpha, iterations):
    """
    Performs gradient descent to optimize the model parameters (theta).

    Parameters:
        x (numpy.ndarray): Input feature matrix (m x n).
        y (numpy.ndarray): Actual target labels (m x 1).
        theta (numpy.ndarray): Initial model parameters (n x 1).
        alpha (float): Learning rate.
        iterations (int): Number of iterations for gradient descent.

    Returns:
        numpy.ndarray: The optimized model parameters (theta).
    """
    m = len(y)
    for i in range(iterations):
        predictions = mu.sigmoid(x @ theta)
        gradient = (1/m) * (x.T @ (predictions - y))
        theta -= alpha * gradient
    return theta


# Preprocessing the dataset
def preprocess_data(filename):
    """
    Preprocesses the dataset by:
    - Dropping irrelevant columns
    - Encoding categorical data
    - Normalizing the features (z-score normalization)

    Parameters:
        filename (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: Tuple containing:
            - numpy.ndarray: Preprocessed feature matrix (X).
            - numpy.ndarray: Target labels (y).
    """
    data_frame = pd.read_csv(filename)
    data_frame = data_frame.drop(columns=['Index', 'First Name',
                                          'Last Name', 'Birthday'])
    data_frame['Best Hand'] = data_frame['Best Hand'].map({'Right': 1,
                                                           'Left': 0})

    # Encode target labels
    houses = {'Gryffindor': 0, 'Hufflepuff': 1,
              'Ravenclaw': 2, 'Slytherin': 3}
    data_frame['Hogwarts House'] = data_frame['Hogwarts House'].map(houses)

    x = data_frame.drop(columns=['Hogwarts House']).values  # Features (scores)
    y = data_frame['Hogwarts House'].values  # Target (house)

    # Normalize each column
    x = pd.DataFrame(x)
    x = x.fillna(0)
    for column in x.columns:
        column_data = x[column].values
        mean_val = mu.mean(column_data)
        std_val = mu.std(column_data, mean_val)
        x[column] = (column_data - mean_val) / std_val
    return x, y


# Train one-vs-all models
def train_ova(x, y, num_classes, alpha, iterations):
    """
    Trains one-vs-all (OVA) logistic regression models for
    multi-class classification.

    Parameters:
        x (numpy.ndarray): Input feature matrix (m x n).
        y (numpy.ndarray): Target labels (m x 1).
        num_classes (int): Number of distinct classes in the target labels.
        alpha (float): Learning rate.
        iterations (int): Number of iterations for gradient descent.

    Returns:
        numpy.ndarray: The trained weights for each class
        (num_classes x (n+1)).
    """
    m, n = x.shape
    weights = np.zeros((num_classes, n+1))  # Add one for bias
    x = np.c_[np.ones((m, 1)), x]

    for c in range(num_classes):
        y_binary = (y == c).astype(int)  # Create binary labels for the class
        theta = np.zeros(x.shape[1])  # Initialize weights
        weights[c] = gradient_descent(x, y_binary, theta, alpha, iterations)
    return weights


def replace_nan_with_mean(x):
    """
    Replaces NaN values in a DataFrame with the column-wise mean.

    Parameters:
        x (pandas.DataFrame): The DataFrame to process.

    Returns:
        pandas.DataFrame: The DataFrame with NaN values
        replaced by column means.
    """
    for i in range(x.shape[1]):  # Iterate through each column
        col = x.iloc[:, i]  # Access column using `.iloc`
        if col.isnull().any():  # Check for NaN values
            mean_value = col.mean()  # Calculate mean of the column
            x.iloc[:, i] = col.fillna(mean_value)  # Replace NaNs with the mean
    return x


# Main function
def main():
    """
    Main function to preprocess data, train a one-vs-all
    logistic regression model, and save the trained weights to a file.

    Expects one command-line argument: the path to the dataset CSV file.
    Outputs:
        A file `weights.npy` containing the trained model weights.
    """
    try:
        if (len(sys.argv) != 2):
            raise AssertionError("wrong number of args")
        print("Preprocessing data...")
        x, y = preprocess_data(sys.argv[1])
        x = replace_nan_with_mean(x)
        print("Training...")
        weights = train_ova(x, y, num_classes=4,
                            alpha=0.001, iterations=10000)
        print("Saving weights...")
        if os.path.exists('weights.npy'):
            os.remove('weights.npy')
        np.save('weights.npy', weights)
        print("Weights saved to weights.npy")
    except Exception as e:
        print(f"logreg_train.py: error: {e}")


if __name__ == "__main__":
    main()
