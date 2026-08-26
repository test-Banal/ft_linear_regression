import json
from pathlib import Path


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
