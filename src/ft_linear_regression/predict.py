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

def main() -> None:
    model_file = (
        Path(__file__).resolve().parents[2]
        / "model"
        / "parameters.json"
    )

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

    print(f"Estimated price: {predicted_price:.2f} €")

if __name__ == "__main__":
    main()