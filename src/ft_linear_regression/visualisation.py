"""Visualise les prix observés et les prédictions de la régression linéaire.

Propose des graphiques et une comparaison chiffrée pour examiner l'ajustement
du modèle. L'exécution du module entraîne le modèle avant de tracer sa droite."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import train


def show_regression(
    data: pd.DataFrame,
    theta0: float,
    theta1: float,
) -> None:
    """Affiche les prix observés et la droite prédite après normalisation du kilométrage."""
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


def compare_predictions(
    data: pd.DataFrame,
    theta0: float,
    theta1: float,
) -> None:
    """Affiche les prix réels et prédits pour chaque voiture, puis les compare en graphique."""
    max_mileage = float(data["km"].max())

    mileage = data["km"]
    real_prices = data["price"]
    normalized_mileage = mileage / max_mileage
    predicted_prices = theta0 + theta1 * normalized_mileage

    for i in range(len(data)):
        print(
            f"{mileage.iloc[i]:.0f} km | "
            f"Real: {real_prices.iloc[i]:.2f} € | "
            f"Predicted: {predicted_prices.iloc[i]:.2f} €"
        )

    sorted_predictions = data.assign(predicted_price=predicted_prices).sort_values("km")

    plt.scatter(
        mileage,
        real_prices,
        color="tab:blue",
        label="Real prices",
        zorder=4,
    )
    plt.scatter(
        mileage,
        predicted_prices,
        color="tab:red",
        label="Predicted prices",
        zorder=3,
    )
    plt.plot(
        sorted_predictions["km"],
        sorted_predictions["predicted_price"],
        color="tab:red",
        linewidth=2,
        label="Our linear regression",
    )

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Real prices vs our predictions")
    plt.legend()
    plt.show()


def main() -> None:
    """Charge le CSV, entraîne le modèle et affiche le graphique de régression."""
    data_file = Path(__file__).resolve().parents[2] / "data" / "data.csv"

    try:
        data = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: data file not found: {data_file}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: data file is empty: {data_file}")
        return
    except pd.errors.ParserError:
        print(f"Error: unable to parse data file: {data_file}")
        return

    train_data = data

    theta0, theta1 = train.train(train_data)

    show_regression(
        train_data,
        theta0,
        theta1,
    )
"""    compare_predictions(
        train_data,
        theta0,
        theta1,
    )"""



if __name__ == "__main__":
    main()
