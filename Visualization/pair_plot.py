"""
pair_plot_hogwarts.py
This script generates a pair plot for the Hogwarts dataset to visualize
relationships between features and decide which ones to use for logistic regression.

Run: python pair_plot_hogwarts.py
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.
    
    :param file_path: Path to the CSV file.
    :return: Loaded DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Hogwarts dataset by removing unnecessary columns
    and handling missing values.
    
    :param data: DataFrame containing the dataset.
    :return: Cleaned DataFrame.
    """
    try:
        # Drop non-numeric and irrelevant columns
        data = data.drop(columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"], errors="ignore")
        
        # Drop rows with missing values or fill them (example: mean)
        data = data.dropna()

        return data
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return pd.DataFrame()

def generate_pair_plot(data: pd.DataFrame, target_column: str = None) -> None:
    """
    Generates a pair plot for the dataset.
    
    :param data: DataFrame containing the dataset.
    :param target_column: Name of the target column.
    """
    try:
        if target_column and target_column in data.columns:
            sns.pairplot(data, hue=target_column, diag_kind="kde", markers=["o", "s", "D", "^"])
        else:
            sns.pairplot(data, diag_kind="kde")
        plt.show()
    except Exception as e:
        print(f"Error generating pair plot: {e}")

def main():
    """
    Main function to execute the pair plot script for the Hogwarts dataset.
    """
    file_path = input("Enter the path to your Hogwarts dataset (CSV file): ")
    target_column = "Hogwarts House"

    data = load_data(file_path)
    if data.empty:
        print("No data to visualize. Exiting.")
        return

    data = clean_data(data)
    if data.empty:
        print("No data to visualize after cleaning. Exiting.")
        return

    generate_pair_plot(data, target_column)

if __name__ == "__main__":
    main()
