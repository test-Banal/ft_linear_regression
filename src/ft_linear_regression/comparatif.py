"""Compare le modele fait maison a des regressions lineaires de bibliotheques."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from .precision import homemade_metrics, predicted_prices
from .predict import load_parameters

ROOT = Path(__file__).resolve().parents[2]


def linear_function(x: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    return intercept + slope * x


def library_predictions(
    mileages: np.ndarray, prices: np.ndarray
) -> dict[str, np.ndarray]:
    sklearn_model = LinearRegression().fit(mileages.reshape(-1, 1), prices)
    polyfit_slope, polyfit_intercept = np.polyfit(mileages, prices, 1)
    scipy_parameters, _ = curve_fit(linear_function, mileages, prices)

    return {
        "sklearn": sklearn_model.predict(mileages.reshape(-1, 1)),
        "numpy.polyfit": linear_function(mileages, polyfit_intercept, polyfit_slope),
        "scipy.curve_fit": linear_function(mileages, *scipy_parameters),
    }


def print_comparison(
    predictions: dict[str, list[float] | np.ndarray], prices: list[float]
) -> None:
    print(f"{'Modele':<20} {'MAE (€)':>12} {'RMSE (€)':>12} {'R2':>10}")
    print("-" * 58)
    for name, values in predictions.items():
        metrics = homemade_metrics(prices, [float(value) for value in values])
        values_text = (
            f"{metrics['mae']:>12.2f} "
            f"{metrics['rmse']:>12.2f} "
            f"{metrics['r2']:>10.6f}"
        )
        print(f"{name:<20} {values_text}")


def plot_comparison(
    mileages: np.ndarray,
    prices: np.ndarray,
    predictions: dict[str, list[float] | np.ndarray],
) -> None:
    order = np.argsort(mileages)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(mileages, prices, color="black", label="Donnees", zorder=3)
    for name, values in predictions.items():
        array = np.asarray(values)
        axes[0].plot(mileages[order], array[order], label=name)
        axes[1].scatter(mileages, prices - array, label=name, alpha=0.75)

    axes[0].set(
        title="Comparaison des regressions",
        xlabel="Kilometrage (km)",
        ylabel="Prix (€)",
    )
    axes[0].legend()
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        title="Comparaison des residus",
        xlabel="Kilometrage (km)",
        ylabel="Reel - predit (€)",
    )
    axes[1].legend()
    figure.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare plusieurs regressions lineaires."
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="N'affiche pas les graphes."
    )
    args = parser.parse_args()

    try:
        data = pd.read_csv(ROOT / "data" / "data.csv")
        theta0, theta1, max_mileage = load_parameters()
    except FileNotFoundError as error:
        print(f"Error: file not found: {error.filename}")
        return
    except (json.JSONDecodeError, KeyError, ValueError) as error:
        print(f"Error: invalid model parameters: {error}")
        return

    if data.empty or not {"km", "price"}.issubset(data.columns):
        print("Error: dataset must contain non-empty 'km' and 'price' columns.")
        return
    if data[["km", "price"]].isna().any().any() or max_mileage <= 0:
        print("Error: dataset or normalization value is invalid.")
        return

    mileages = data["km"].to_numpy(dtype=float)
    prices_array = data["price"].to_numpy(dtype=float)
    prices = [float(value) for value in prices_array]
    predictions: dict[str, list[float] | np.ndarray] = {
        "Notre modele": predicted_prices(data["km"], theta0, theta1, max_mileage)
    }
    predictions.update(library_predictions(mileages, prices_array))

    print_comparison(predictions, prices)
    if not args.no_plot:
        plot_comparison(mileages, prices_array, predictions)


if __name__ == "__main__":
    main()
