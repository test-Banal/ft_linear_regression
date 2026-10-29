"""BONUS BONUS 
A ENLEVER

Complète l'évaluation du modèle maison par une analyse des erreurs.

Compare ses performances à une prédiction constante égale au prix moyen.
Ajoute l'erreur médiane, l'erreur maximale et des graphiques de diagnostic."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .precision import homemade_metrics, predicted_prices
from .predict import load_parameters

ROOT = Path(__file__).resolve().parents[2]


def median_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    """Renvoie la médiane des erreurs absolues de prix, en euros."""
    if not y_true:
        raise ValueError("Cannot compute an error on an empty dataset.")
    errors = sorted(
        abs(real - predicted)
        for real, predicted in zip(y_true, y_pred, strict=True)
    )
    middle = len(errors) // 2
    if len(errors) % 2:
        return errors[middle]
    return (errors[middle - 1] + errors[middle]) / 2


def max_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    """Renvoie la plus grande erreur absolue de prix, en euros."""
    if not y_true:
        raise ValueError("Cannot compute an error on an empty dataset.")
    return max(
        abs(real - predicted)
        for real, predicted in zip(y_true, y_pred, strict=True)
    )


def mean_baseline(y_true: list[float]) -> list[float]:
    """Construit une prédiction de référence égale au prix moyen pour chaque observation."""
    if not y_true:
        raise ValueError("Cannot build a baseline on an empty dataset.")
    mean_price = sum(y_true) / len(y_true)
    return [mean_price for _ in y_true]


def plot_precision(
    mileages: list[float],
    y_true: list[float],
    y_pred: list[float],
    baseline_pred: list[float],
) -> None:
    """Trace les résidus du modèle et compare ses erreurs à celles de la référence moyenne."""
    model_metrics = homemade_metrics(y_true, y_pred)
    baseline_metrics = homemade_metrics(y_true, baseline_pred)
    model_euro_metrics = [
        model_metrics["mae"],
        median_absolute_error(y_true, y_pred),
        model_metrics["rmse"],
        max_absolute_error(y_true, y_pred),
    ]
    baseline_euro_metrics = [
        baseline_metrics["mae"],
        median_absolute_error(y_true, baseline_pred),
        baseline_metrics["rmse"],
        max_absolute_error(y_true, baseline_pred),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    residuals = [
        real - predicted
        for real, predicted in zip(y_true, y_pred, strict=True)
    ]
    axes[0].scatter(mileages, residuals, color="tab:blue")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(
        title="Residus du modele", xlabel="Kilometrage (km)", ylabel="Reel - predit (€)"
    )

    labels = ["MAE", "Mediane", "RMSE", "Erreur max"]
    positions = list(range(len(labels)))
    width = 0.38
    axes[1].bar(
        [p - width / 2 for p in positions],
        model_euro_metrics,
        width,
        label="Notre modele",
    )
    axes[1].bar(
        [p + width / 2 for p in positions],
        baseline_euro_metrics,
        width,
        label="Baseline moyenne",
    )
    axes[1].set_xticks(positions, labels)
    axes[1].set(title="Erreurs du modele et de la baseline", ylabel="Erreur (€)")
    axes[1].legend()
    figure.tight_layout()
    plt.show()


def main() -> None:
    """Évalue le modèle et la référence moyenne, puis affiche les graphiques sauf avec --no-plot."""
    parser = argparse.ArgumentParser(description="Mesure la precision du modele.")
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

    mileages = [float(value) for value in data["km"]]
    y_true = [float(value) for value in data["price"]]
    y_pred = predicted_prices(data["km"], theta0, theta1, max_mileage)
    baseline_pred = mean_baseline(y_true)
    model = homemade_metrics(y_true, y_pred)
    baseline = homemade_metrics(y_true, baseline_pred)

    print("Precision de notre modele")
    print(f"  MAE                    : {model['mae']:.2f} €")
    print(f"  Mediane erreur absolue : {median_absolute_error(y_true, y_pred):.2f} €")
    print(f"  RMSE                   : {model['rmse']:.2f} €")
    print(f"  Erreur maximale        : {max_absolute_error(y_true, y_pred):.2f} €")
    print(f"  R2                     : {model['r2']:.4f}")
    print("Baseline (prediction constante = prix moyen)")
    print(f"  MAE                    : {baseline['mae']:.2f} €")
    print(f"  RMSE                   : {baseline['rmse']:.2f} €")
    improvement = 100 * (baseline["mae"] - model["mae"]) / baseline["mae"]
    print(f"  Amelioration MAE       : {improvement:.2f} %")

    if not args.no_plot:
        plot_precision(mileages, y_true, y_pred, baseline_pred)


if __name__ == "__main__":
    main()
