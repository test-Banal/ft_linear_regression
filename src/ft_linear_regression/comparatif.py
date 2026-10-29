"""
A enlever pour git

Compare le modèle maison aux régressions de scikit-learn, NumPy et SciPy.

Évalue leurs prédictions sur les mêmes données avec MAE, RMSE et R², puis
propose des graphiques des droites ajustées et des résidus."""

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
    """Évalue la droite intercept + slope × x pour les valeurs fournies."""
    return intercept + slope * x


def library_predictions(
    mileages: np.ndarray, prices: np.ndarray
) -> dict[str, np.ndarray]:
    """Ajuste trois régressions de bibliothèques et renvoie leurs prédictions sur les données."""
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
    """Affiche un tableau des scores MAE, RMSE et R² pour chaque modèle."""
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
    """Trace les droites de régression et les résidus de chaque modèle."""
    order = np.argsort(mileages)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(mileages, prices, color="black", label="Donnees", zorder=3)

    colors = {
        "Notre modele": "tab:blue",
        "sklearn": "tab:orange",
        "numpy.polyfit": "tab:green",
        "scipy.curve_fit": "tab:red",
    }
    line_styles = {
        "Notre modele": "-",
        "sklearn": "--",
        "numpy.polyfit": ":",
        "scipy.curve_fit": "-.",
    }
    line_widths = {
        "Notre modele": 4.0,
        "sklearn": 3.0,
        "numpy.polyfit": 2.5,
        "scipy.curve_fit": 1.8,
    }
    markers = {
        "Notre modele": "o",
        "sklearn": "s",
        "numpy.polyfit": "^",
        "scipy.curve_fit": "D",
    }
    marker_sizes = {
        "Notre modele": 110,
        "sklearn": 75,
        "numpy.polyfit": 45,
        "scipy.curve_fit": 20,
    }

    for name, values in predictions.items():
        array = np.asarray(values)
        axes[0].plot(
            mileages[order],
            array[order],
            color=colors[name],
            linestyle=line_styles[name],
            linewidth=line_widths[name],
            label=name,
        )
        axes[1].scatter(
            mileages,
            prices - array,
            color=colors[name],
            marker=markers[name],
            s=marker_sizes[name],
            label=name,
            alpha=0.85,
        )

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
    """Charge les données et le modèle, puis lance la comparaison avec un graphique optionnel."""
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
