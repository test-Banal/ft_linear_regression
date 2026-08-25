import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from train import training
from train import train_step
from predict import estimate_price

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


def main() -> None:
    # resolve() supprime les segments relatifs comme "../.." du chemin.
    data_file = Path(__file__).resolve().parents[2] / "data" / "data.csv"
    data = pd.read_csv(data_file)
    train_data = data
    #test_info(data)
    #test_show(data)
    #print_fin(data)
    #training(train_data)
    theta0 = 0.0
    theta1 = 0.0
    learning_rate = 0.1
    iterations = 1000

    for i in range(iterations):
        theta0, theta1 = train_step(train_data, theta0, theta1, learning_rate)

    print(f"theta0: {theta0}")
    print(f"theta1: {theta1}")

    test_resulatl = estimate_price()

if __name__ == "__main__":
    main()
