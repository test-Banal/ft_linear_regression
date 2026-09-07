import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
from . import train

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
    predicted_prices = theta0 + theta1 * normalized_mileage

    # sklearn attend une matrice avec une colonne pour les donnees d'entree.
    mileage_array = mileage.to_numpy(dtype=float)
    real_prices_array = real_prices.to_numpy(dtype=float)
    sklearn_model = LinearRegression()
    sklearn_model.fit(mileage_array.reshape(-1, 1), real_prices_array)
    sklearn_prices = sklearn_model.predict(mileage_array.reshape(-1, 1))

    # polyfit retourne d'abord la pente, puis l'ordonnee a l'origine.
    numpy_slope, numpy_intercept = np.polyfit(
        mileage_array,
        real_prices_array,
        1,
    )
    numpy_prices = numpy_intercept + numpy_slope * mileage_array

    for i in range(len(data)):
        print(
            f"{mileage.iloc[i]:.0f} km | "
            f"Real: {real_prices.iloc[i]:.2f} € | "
            f"Mine: {predicted_prices.iloc[i]:.2f} € | "
            f"sklearn: {sklearn_prices[i]:.2f} € | "
            f"polyfit: {numpy_prices[i]:.2f} €"
        )

    # Trier les x garantit que les droites sont tracees de gauche a droite.
    order = np.argsort(mileage_array)

    plt.scatter(
        mileage,
        real_prices,
        color="black",
        label="Real prices",
        zorder=4,
    )
    plt.plot(
        mileage_array[order],
        predicted_prices.to_numpy()[order],
        color="tab:blue",
        linewidth=3,
        label="Our gradient descent",
    )
    plt.plot(
        mileage_array[order],
        sklearn_prices[order],
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        label="sklearn LinearRegression",
    )
    plt.plot(
        mileage_array[order],
        numpy_prices[order],
        color="tab:green",
        linestyle=":",
        linewidth=3,
        label="numpy.polyfit",
    )

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Real prices vs regression models")
    plt.legend()
    plt.show()

def main() -> None:
    data_file = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "data.csv"
    )

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

    visualisation.compare_predictions(
        train_data,
        theta0,
        theta1,
    )


if __name__ == "__main__":
    main()
