import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def estimate_price(
    mileage: float,
    theta0: float,
    theta1: float,
) -> float:
    return theta0 + theta1 * mileage


def load_parameters() -> tuple[float, float, float]:
    model_file = (
        Path(__file__).resolve().parents[2]
        / "model"
        / "parameters.json"
    )

    with model_file.open("r") as file:
        parameters = json.load(file)

    theta0 = float(parameters["theta0"])
    theta1 = float(parameters["theta1"])
    max_mileage = float(parameters["max_mileage"])

    return theta0, theta1, max_mileage


def library_estimates(data: pd.DataFrame, mileage: float) -> tuple[float, float]:
    """Calcule les estimations de sklearn et numpy pour comparaison."""
    mileages = data["km"].to_numpy(dtype=float)
    prices = data["price"].to_numpy(dtype=float)

    sklearn_model = LinearRegression()
    sklearn_model.fit(mileages.reshape(-1, 1), prices)
    sklearn_price = float(sklearn_model.predict([[mileage]])[0])

    numpy_slope, numpy_intercept = np.polyfit(mileages, prices, 1)
    numpy_price = float(numpy_intercept + numpy_slope * mileage)
    return sklearn_price, numpy_price


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    model_file = root / "model" / "parameters.json"
    data_file = root / "data" / "data.csv"

    if not model_file.exists():
        print("Error: model has not been trained yet.")
        return

    try:
        theta0, theta1, max_mileage = load_parameters()
    except json.JSONDecodeError:
        print("Error: model parameters file is corrupted.")
        return
    except KeyError:
        print("Error: model parameters are incomplete.")
        return

    if max_mileage <= 0:
        print("Error: invalid model normalization value.")
        return

    try:
        data = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: data file not found: {data_file}")
        return
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        print(f"Error: invalid data file: {data_file}")
        return

    if data.empty or not {"km", "price"}.issubset(data.columns):
        print("Error: dataset must contain non-empty 'km' and 'price' columns.")
        return
    if data[["km", "price"]].isna().any().any():
        print("Error: dataset contains missing values.")
        return

    try:
        mileage = float(input("Mileage: "))
    except ValueError:
        print("Error: mileage must be a number.")
        return

    if mileage < 0:
        print("Error: mileage cannot be negative.")
        return

    normalized_mileage = mileage / max_mileage

    predicted_price = estimate_price(
        normalized_mileage,
        theta0,
        theta1,
    )
    sklearn_price, numpy_price = library_estimates(data, mileage)

    print(f"Estimated price: {predicted_price:.2f} €")
    print(f"Estimated price with sklearn: {sklearn_price:.2f} €")
    print(f"Estimated price with numpy.polyfit: {numpy_price:.2f} €")


if __name__ == "__main__":
    main()
