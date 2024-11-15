# describe.py
import pandas as pd
import sys

def calculate_count(data):
    return sum(1 for x in data if not pd.isnull(x))

def calculate_mean(data):
    count = calculate_count(data)
    return sum(x for x in data if not pd.isnull(x)) / count if count > 0 else None

def calculate_std(data, mean):
    count = calculate_count(data)
    if count <= 1:
        return None
    variance = sum((x - mean) ** 2 for x in data if not pd.isnull(x)) / (count - 1)
    return variance ** 0.5

def calculate_min(data):
    return min(x for x in data if not pd.isnull(x))

def calculate_max(data):
    return max(x for x in data if not pd.isnull(x))

def calculate_percentile(data, percentile):
    sorted_data = sorted(x for x in data if not pd.isnull(x))
    index = (len(sorted_data) - 1) * percentile
    lower = int(index)
    upper = lower + 1
    weight = index - lower
    if upper >= len(sorted_data):
        return sorted_data[lower]
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

def describe(dataframe):
    numeric_data = dataframe.select_dtypes(include=['float64', 'int64'])
    stats = {
        'Feature': [],
        'Count': [],
        'Mean': [],
        'Std': [],
        'Min': [],
        '25%': [],
        '50%': [],
        '75%': [],
        'Max': []
    }

    for column in numeric_data.columns:
        data = numeric_data[column].tolist()
        count = calculate_count(data)
        mean = calculate_mean(data)
        std = calculate_std(data, mean)
        minimum = calculate_min(data)
        percentile_25 = calculate_percentile(data, 0.25)
        median = calculate_percentile(data, 0.5)
        percentile_75 = calculate_percentile(data, 0.75)
        maximum = calculate_max(data)

        stats['Feature'].append(column)
        stats['Count'].append(count)
        stats['Mean'].append(mean)
        stats['Std'].append(std)
        stats['Min'].append(minimum)
        stats['25%'].append(percentile_25)
        stats['50%'].append(median)
        stats['75%'].append(percentile_75)
        stats['Max'].append(maximum)

    # Convertir les statistiques en DataFrame
    stats_df = pd.DataFrame(stats)
    stats_df.set_index('Feature', inplace=True)

    return stats_df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Erreur : Fichier '{file_path}' introuvable.")
        sys.exit(1)

    # Générer et afficher les statistiques descriptives
    stats = describe(data)
    print(stats.to_string())
