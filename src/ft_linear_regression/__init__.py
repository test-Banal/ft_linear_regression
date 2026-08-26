import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from train import training
from train import train_step
from train import training_exceptation
from predict import estimate_price
from train import cost_function
from train import save_parameters

def compare_predictions(
    data: pd.DataFrame,
    theta0: float,
    theta1: float,
) -> None:
    max_mileage = float(data["km"].max())

    mileage = data["km"]
    real_prices = data["price"]

    normalized_mileage = mileage / max_mileage

    predicted_prices = (
        theta0 + theta1 * normalized_mileage
    )
    for i in range(len(data)):
        print(
            f"{mileage.iloc[i]:.0f} km | "
            f"Real: {real_prices.iloc[i]:.2f} € | "
            f"Predicted: {predicted_prices.iloc[i]:.2f} €"
        )
    
    plt.scatter(
        mileage,
        real_prices,
        color="blue",
        label="Real prices",
    )

    plt.scatter(
        mileage,
        predicted_prices,
        color="red",
        label="Predicted prices",
    )

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Real prices vs predicted prices")
    plt.legend()
    plt.show()


def show_regression(
    data: pd.DataFrame,
    theta0: float,
    theta1: float,
) -> None:
    max_mileage = float(data["km"].max())

    mileage = data["km"]
    normalized_mileage = mileage / max_mileage

    predicted_prices = theta0 + theta1 * normalized_mileage

    plt.scatter(
        mileage,
        data["price"],
        label="Real data",
    )

    plt.plot(
        mileage,
        predicted_prices,
        label="Linear regression",
    )

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Price according to mileage")
    plt.legend()
    plt.show()


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

    theta0, theta1 = training_exceptation(train_data)

    compare_predictions(
        train_data,
        theta0,
        theta1,
    )   



    show_regression(
        train_data,
        theta0,
        theta1,
    )
    print( 
        f"theta0={theta0:.2f}, "
        f"theta1={theta1:.2f}"
    )
    max_mileage = float(train_data["km"].max())
    save_parameters(
        theta0,
        theta1,
        max_mileage,
)


if __name__ == "__main__":
    main()
