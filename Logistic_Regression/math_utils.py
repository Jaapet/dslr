import numpy as np


def mean(arr):
    """
    Computes the mean of a list of numbers.

    Parameters:
        arr (list of float): The list of numbers.

    Returns:
        float: The mean of the numbers in the list.
    """
    total_sum = 0
    for val in arr:
        total_sum += val
    return total_sum / len(arr)


def std(arr, mean_val):
    """
    Computes the standard deviation of a list of numbers,
    given the mean.

    Parameters:
        arr (list of float): The list of numbers.
        mean_val (float): The mean of the numbers.

    Returns:
        float: The standard deviation of the numbers.
    """
    variance_sum = 0
    for val in arr:
        variance_sum += (val - mean_val) ** 2
    variance = variance_sum / len(arr)
    return variance ** 0.5


def sigmoid(x):
    """
    Computes the sigmoid function.

    Parameters:
        x (float or numpy.ndarray): The input value(s).

    Returns:
        float or numpy.ndarray: The sigmoid of the input.
    """
    return 1 / (1 + np.exp(-x))


def argmax(array):
    """
    Returns the indices of the maximum value
    in each row of a 2D array.

    Parameters:
        array (numpy.ndarray): The 2D array.

    Returns:
        list of int: The indices of the maximum values
        for each row.
    """
    max_indices = []
    for row in range(array.shape[0]):
        row_values = array[row, :]
        max_value = row_values[0]
        max_index = 0
        for i in range(1, len(row_values)):
            if row_values[i] > max_value:
                max_value = row_values[i]
                max_index = i
        max_indices.append(max_index)
    return max_indices
