import pandas as pd
import matplotlib.pyplot as plt



def test_info(data: pd.DataFrame) -> None:
    """Affiche les informations utiles pour l'EDA et le contrôle des données."""
    print(data)  # Utiliser data.head() pour n'afficher que les premières lignes.
    print(data.shape)
    print()
    print(data.columns)
    data.info()
    print()
    print("Valeurs manquantes")
    print(data.isna().sum())
    print("Doublons :")
    print(data.duplicated().sum())
    print("Valeurs km anormales")
    print(data[data["km"] < 0])
    print("Valeurs prix anormales")
    print(data[data["price"] < 0])
    print("Informations supplémentaires")
    print(data.describe())
    print()


def test_show(data: pd.DataFrame) -> None:
    """Affiche le graphique du prix en fonction du kilométrage."""
    plt.scatter(data["km"], data["price"])
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Price according to mileage")
    plt.show()


def print_fin(data: pd.DataFrame) -> None:
    """Affiche la corrélation et les données triées par kilométrage."""
    print("Corrélation :")
    print(data.corr(numeric_only=True))
    print()
    print("Données triées par kilométrage :")
    df_sorted = data.sort_values(by="km")
    print(df_sorted)
