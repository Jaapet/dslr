import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def scatter_plot(file_path):
    """
    Generates and displays a scatter plot for
    the two most correlated numeric features
    in the dataset, colored by Hogwarts House.

    Parameters:
        file_path (str): The path to the CSV file containing the dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If there are not enough numeric features
        to generate a scatter plot.
    """
    try:
        # Charger les données
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Erreur : Fichier '{file_path}' introuvable.")
        sys.exit(1)

    # Exclure la colonne 'Index'
    # (et toute autre colonne non pertinente comme
    # 'First Name', 'Last Name', etc.)
    data = data.drop(columns=['Index', 'First Name',
                              'Last Name', 'Birthday', 'Best Hand'])

    # Sélectionner les colonnes numériques restantes
    numeric_features = data.select_dtypes(include=['float64', 'int64'])\
        .columns.tolist()

    if len(numeric_features) < 2:
        print("Erreur : Pas assez de features numériques " +
              "pour tracer un scatter plot.")
        sys.exit(1)

    # Identifier les deux features les plus corrélées
    # parmi des paires distinctes
    corr_matrix = data[numeric_features].corr()

    # Masquer la corrélation d'une feature avec elle-même (diagonale)
    abs_corr = corr_matrix.abs()

    # Remplacer les valeurs sur la diagonale par des valeurs
    # non significatives (par exemple, 0 ou NaN)
    for i in range(len(abs_corr)):
        abs_corr.iloc[i, i] = 0

    # Trouver la paire de features avec la plus grande corrélation
    # (parmi des paires distinctes)
    max_corr = abs_corr.unstack().sort_values(ascending=False).idxmax()

    # Assurer que les deux features sont distinctes
    feature_x, feature_y = max_corr
    # Si les features sont identiques, chercher une autre paire
    while feature_x == feature_y:
        # Remettre à 0 la corrélation entre ces deux
        abs_corr.iloc[feature_x, feature_y] = 0
        max_corr = abs_corr.unstack().sort_values(ascending=False).idxmax()
        feature_x, feature_y = max_corr

    print("Les deux features les plus corrélées " +
          f"et distinctes sont : '{feature_x}' et '{feature_y}'.")

    # Tracer le scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x=feature_x,
        y=feature_y,
        hue="Hogwarts House",
        palette="bright",
        alpha=0.7
    )
    plt.title(f"Scatter Plot : {feature_x} vs {feature_y}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend(title="Hogwarts House",
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Afficher le graphique
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python scatter_plot.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    scatter_plot(file_path)
