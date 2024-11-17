import numpy as np
import pandas as pd
import sys
import os
import math_utils as mu


# Preprocessing the test dataset
def preprocess_test_data(filename):
    data_frame = pd.read_csv(filename)
    indices = data_frame['Index']
    data_frame = data_frame.drop(columns=['Index', 'First Name',
                                          'Last Name', 'Birthday'])
    data_frame['Best Hand'] = data_frame['Best Hand'].map({'Right': 1,
                                                           'Left': 0})
    
    x = data_frame.drop(columns=['Hogwarts House']).values
    x = pd.DataFrame(x)
    for column in x.columns:
        column_data = x[column].values
        mean_val = mu.mean(column_data)
        std_val = mu.std(column_data, mean_val)
        x[column] = (column_data - mean_val) / std_val

    return x, indices


# Predict function
def predict(x, weights):
    x = np.c_[np.ones((x.shape[0], 1)), x]  # Add bias term
    probabilities = mu.sigmoid(x @ weights.T)
    # print("Probabilities:\n", probabilities[:5])
    return mu.argmax(probabilities)


# Main function
def main():
    # try:
        if len(sys.argv) != 3:
            raise AssertionError("wrong number of args")
        x, indices = preprocess_test_data(sys.argv[1])
        weights = np.load(sys.argv[2])
        print("Weights:", weights)
        predictions = predict(x, weights)
        
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        predictions_houses = [houses[p] for p in predictions]
        
        output = pd.DataFrame({'Index': indices, 'Hogwarts House': predictions_houses})
        if os.path.exists('houses.csv'):
            os.remove('houses.csv')
        output.to_csv('houses.csv', index=False)
        print("Predictions saved to houses.csv")
    # except Exception as e:
    #     print(f"logreg_test.py: error: {e}")


if __name__ == "__main__":
    main()
