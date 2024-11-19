import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def plot_histograms(file_path):
    """
    Plots histograms for each course's score distribution,
    colored by Hogwarts House.

    Parameters:
        file_path (str): The path to the CSV file containing the dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the 'Hogwarts House' column is missing
        or if all course data is NaN.
    """
    try:
        # Charger les données
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Erreur : Fichier '{file_path}' introuvable.")
        sys.exit(1)

    # Vérifier si 'Hogwarts House' est présente
    if 'Hogwarts House' not in data.columns:
        print("Erreur : La colonne 'Hogwarts House' est absente.")
        sys.exit(1)

    # Supprimer les lignes avec des maisons manquantes
    data = data.dropna(subset=['Hogwarts House'])

    # Sélectionner les colonnes des cours (ignorer les colonnes non numériques)
    courses = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Définir le nombre de colonnes et de lignes pour l'affichage
    num_courses = len(courses)
    cols = 3  # Nombre de colonnes par ligne
    rows = (num_courses + cols - 1) // cols  # Calcul du nombre de lignes

    # Créer une figure avec sous-graphiques
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Aplatir la grille pour un accès facile

    # Générer un histogramme pour chaque cours
    for i, course in enumerate(courses):
        # Vérifier si le cours a des valeurs exploitables
        if data[course].isnull().all():
            continue

        # Créer l'histogramme
        sns.histplot(
            data=data,
            x=course,
            hue="Hogwarts House",
            element="step",
            stat="density",
            common_norm=False,
            ax=axes[i],
            palette="bright"
        )
        axes[i].set_title(f"{course}")
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Densité")

        # Déplacer la légende à l'extérieur (sur la droite)
        handles, labels = axes[i].get_legend_handles_labels()
        if handles:  # S'assurer que la légende contient des éléments
            axes[i].legend(
                handles=handles,
                labels=labels,
                title="Hogwarts House",
                loc="center left",  # Position au centre gauche
                bbox_to_anchor=(1.0, 0.5),  # En dehors du graphique
                borderaxespad=0.5,  # Ajustement de l'espacement
            )

    # Supprimer les sous-graphiques inutilisés (si rows * cols > num_courses)
    for j in range(num_courses, len(axes)):
        fig.delaxes(axes[j])

    # Ajouter du padding entre les graphiques
    plt.tight_layout(pad=3.0)

    # Ajuster les marges pour permettre de la place aux légendes
    fig.subplots_adjust(right=0.85)  # Réserver de l'espace sur la droite

    # Afficher les graphiques
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python histogram.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    plot_histograms(file_path)
