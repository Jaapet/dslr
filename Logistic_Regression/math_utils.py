import numpy as np


def mean(arr):
    total_sum = 0
    for val in arr:
        if np.isnan(val):
            continue
        total_sum += val
    return total_sum / len(arr)


def std(arr, mean_val):
    variance_sum = 0
    for val in arr:
        if np.isnan(val):
            continue
        variance_sum += (val - mean_val) ** 2
    variance = variance_sum / len(arr)
    return variance ** 0.5


def sigmoid(x):
    # x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def argmax(array):
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
