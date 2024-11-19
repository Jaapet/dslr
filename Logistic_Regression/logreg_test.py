import numpy as np
import pandas as pd
import sys
import os
import math_utils as mu


# Preprocessing the test dataset
def preprocess_test_data(filename):
    """
    Preprocesses the test dataset by:
    - Dropping irrelevant columns
    - Mapping categorical data to numeric values
    - Filling missing values with 0
    - Normalizing the features using z-score normalization

    Parameters:
        filename (str): Path to the CSV file containing the test data.

    Returns:
        numpy.ndarray: Preprocessed feature matrix with added bias term.
    """
    data_frame = pd.read_csv(filename)

    features = data_frame.drop(columns=['Index', 'Hogwarts House',
                                        'First Name', 'Last Name', 'Birthday'])
    features['Best Hand'] = features['Best Hand'].map({'Right': 1,
                                                       'Left': 0})
    features = features.fillna(0)  # Handle NaNs

    for column in features.columns:
        column_data = features[column].values
        mean_val = mu.mean(column_data)
        std_val = mu.std(column_data, mean_val)
        features[column] = (column_data - mean_val) / std_val

    m, n = features.shape
    features = np.c_[np.ones((m, 1)), features]

    return features


# Predict function
def predict(features, weights):
    """
    Predicts the class labels by calculating probabilities
    and selecting the class with the highest probability.

    Parameters:
        features (numpy.ndarray): The preprocessed feature matrix.
        weights (numpy.ndarray): The weights from the trained
                                 logistic regression model.

    Returns:
        numpy.ndarray: Array of predicted class labels
        (indices of highest probability).
    """
    # Predict probabilities
    probabilities = mu.sigmoid(np.dot(features, weights.T))
    # Choose the class with the highest probability
    return np.argmax(probabilities, axis=1)


# Main function
def main():
    """
    Main function to preprocess the test data,
    load the trained model weights,
    make predictions, and save the results to a CSV file.

    The program expects two command-line arguments:
    - The path to the test data CSV file.
    - The path to the trained weights file (numpy format).

    Outputs:
        A CSV file `houses.csv` containing the predicted
        Hogwarts houses for each test entry.
    """
    try:
        if len(sys.argv) != 3:
            raise AssertionError("wrong number of args")
        features = preprocess_test_data(sys.argv[1])
        weights = np.load(sys.argv[2])
        predictions = predict(features, weights)

        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        predictions_houses = [houses[p] for p in predictions]

        test_data = pd.read_csv(sys.argv[1])
        indices = test_data['Index']

        output = pd.DataFrame({'Index': indices,
                               'Hogwarts House': predictions_houses})
        if os.path.exists('houses.csv'):
            os.remove('houses.csv')
        output.to_csv('houses.csv', index=False)
        print("Predictions saved to houses.csv")
    except Exception as e:
        print(f"logreg_test.py: error: {e}")


if __name__ == "__main__":
    main()
