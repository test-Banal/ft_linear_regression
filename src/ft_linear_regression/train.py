import pandas as pd
from pathlib import Path
from predict import estimate_price

def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
        return theta0 + theta1 * mileage


def training(train_data: pd.DataFrame) -> None:
    mileage = train_data["km"]
    price = train_data["price"]

    theta0 = 0.0
    theta1 = 0.0

    prediction = estimate_price(
        mileage = float(mileage.iloc[0]),
        theta0=theta0,
        theta1=theta1,
    )

    real_price = float(price.iloc[0])
    error = prediction - real_price

    print(f"Mileage: {mileage.iloc[0]}")
    print(f"Real price: {price.iloc[0]}")
    print(f"Predicted price: {prediction}")
    print(f"Error : {error}")