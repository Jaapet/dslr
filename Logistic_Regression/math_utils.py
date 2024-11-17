def len_set(set):
    len = 0
    for e in set:
        len += 1
    return len


def sum_set(set):
    sum = 0
    for e in set:
        sum += e
    return sum


def mean(set):
    return sum_set(set) / len_set(set)


def std(set, mean):
    return (sum_set((x - mean) ** 2 for x in set) / len_set(set)) ** 0.5


def abs(x):
    return x if x >= 0 else -x


def exp(x, terms=1000):
    result = 1
    term = 1
    for i in range(1, terms):
        term *= x / i
        result += term
        if abs(term) < 1e-15:
            break
    return result


def log(x, tolerance=1e-7):
    if x <= 0:
        raise ValueError("Logarithm is undefined for non-positive values.")

    y = 0  # Initial guess
    while abs(exp(y) - x) > tolerance:
        y -= (exp(y) - x) / exp(y)  # Newton-Raphson update
    return y
