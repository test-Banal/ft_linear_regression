import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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

    plt.plot(
        mileage,
        predicted_prices,
        label="Linear regression",
    )

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Real prices vs predicted prices")
    plt.legend()
    plt.show()