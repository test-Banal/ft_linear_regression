import json
from math import sqrt
from pathlib import Path

import pandas as pd

from .predict import estimate_price, load_parameters

METRIC_NAMES = ("mae", "mse", "rmse", "r2")


def _as_floats(values: pd.Series) -> list[float]:
    return [float(value) for value in values]


def predicted_prices(
    mileages: pd.Series,
    theta0: float,
    theta1: float,
    max_mileage: float,
) -> list[float]:
    """Apply the trained model to each mileage (same normalization as predict)."""
    return [
        estimate_price(float(mileage) / max_mileage, theta0, theta1)
        for mileage in mileages
    ]


def mean_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    """MAE: average absolute error, in the same unit as price (€)."""
    m = len(y_true)
    if m == 0:
        raise ValueError("Cannot compute MAE on an empty dataset.")
    total = 0.0
    for i in range(m):
        total += abs(y_pred[i] - y_true[i])
    return total / m


def mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
    """MSE: average squared error. Penalizes large mistakes more than MAE."""
    m = len(y_true)
    if m == 0:
        raise ValueError("Cannot compute MSE on an empty dataset.")
    total = 0.0
    for i in range(m):
        error = y_pred[i] - y_true[i]
        total += error ** 2
    return total / m


def root_mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
    """RMSE: square root of MSE, back in €."""
    return sqrt(mean_squared_error(y_true, y_pred))


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    """R²: 1 if predictions match perfectly, 0 if no better than predicting the mean."""
    m = len(y_true)
    if m == 0:
        raise ValueError("Cannot compute R² on an empty dataset.")

    mean_true = sum(y_true) / m
    residual_sum = 0.0
    total_sum = 0.0
    for i in range(m):
        residual_sum += (y_true[i] - y_pred[i]) ** 2
        total_sum += (y_true[i] - mean_true) ** 2

    if total_sum == 0.0:
        return 1.0 if residual_sum == 0.0 else 0.0
    return 1.0 - (residual_sum / total_sum)


def homemade_metrics(
    y_true: list[float],
    y_pred: list[float],
) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def sklearn_metrics(
    y_true: list[float],
    y_pred: list[float],
) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error as sk_mae
    from sklearn.metrics import mean_squared_error as sk_mse
    from sklearn.metrics import r2_score as sk_r2
    from sklearn.metrics import root_mean_squared_error as sk_rmse

    return {
        "mae": float(sk_mae(y_true, y_pred)),
        "mse": float(sk_mse(y_true, y_pred)),
        "rmse": float(sk_rmse(y_true, y_pred)),
        "r2": float(sk_r2(y_true, y_pred)),
    }


def sklearn_linear_regression_prices(train_data: pd.DataFrame) -> list[float]:
    """OLS fit from scikit-learn — comparison only, not used to train our model."""
    from sklearn.linear_model import LinearRegression

    mileages = train_data[["km"]].to_numpy()
    prices = train_data["price"].to_numpy()
    model = LinearRegression()
    model.fit(mileages, prices)
    predicted = model.predict(mileages)
    return [float(price) for price in predicted]


def print_metrics_block(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print(f"  MAE  : {metrics['mae']:.8f} €")
    print(f"  MSE  : {metrics['mse']:.8f}")
    print(f"  RMSE : {metrics['rmse']:.8f} €")
    print(f"  R2   : {metrics['r2']:.8f}")


def print_deltas(
    homemade: dict[str, float],
    reference: dict[str, float],
) -> None:
    print("  Delta (homemade - sklearn.metrics)")
    for name in METRIC_NAMES:
        print(f"    {name.upper():<4}: {homemade[name] - reference[name]:.2e}")


def main() -> None:
    data_file = Path(__file__).resolve().parents[2] / "data" / "data.csv"
    model_file = Path(__file__).resolve().parents[2] / "model" / "parameters.json"

    if not model_file.exists():
        print("Error: model has not been trained yet.")
        return

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

    required_columns = {"km", "price"}
    if not required_columns.issubset(data.columns):
        print("Error: dataset must contain 'km' and 'price' columns.")
        return

    if data.empty:
        print("Error: dataset is empty.")
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

    y_true = _as_floats(data["price"])
    our_predictions = predicted_prices(
        data["km"],
        theta0,
        theta1,
        max_mileage,
    )
    sklearn_predictions = sklearn_linear_regression_prices(data)

    # 1) Our formulas, our predictions vs data.csv prices
    homemade_on_ours = homemade_metrics(y_true, our_predictions)
    # 2) sklearn.metrics, our predictions vs data.csv prices
    sklearn_on_ours = sklearn_metrics(y_true, our_predictions)
    # 3) Our formulas, sklearn LinearRegression vs data.csv prices
    homemade_on_sklearn_model = homemade_metrics(y_true, sklearn_predictions)
    # 4) sklearn.metrics, sklearn LinearRegression vs data.csv prices
    sklearn_on_sklearn_model = sklearn_metrics(y_true, sklearn_predictions)

    print("y_true  = prices from data/data.csv")
    print("y_pred  = either our gradient-descent model, or sklearn LinearRegression")
    print()
    print_metrics_block(
        "1) Homemade metrics | our predict vs data prices",
        homemade_on_ours,
    )
    print()
    print_metrics_block(
        "2) sklearn.metrics  | our predict vs data prices",
        sklearn_on_ours,
    )
    print_deltas(homemade_on_ours, sklearn_on_ours)
    print()
    print_metrics_block(
        "3) Homemade metrics | sklearn LinearRegression vs data prices",
        homemade_on_sklearn_model,
    )
    print()
    print_metrics_block(
        "4) sklearn.metrics  | sklearn LinearRegression vs data prices",
        sklearn_on_sklearn_model,
    )
    print_deltas(homemade_on_sklearn_model, sklearn_on_sklearn_model)


if __name__ == "__main__":
    main()
