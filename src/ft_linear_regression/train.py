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

def train_step(train_data: pd.DataFrame, theta0: float, theta1: float, learning_rate:0) -> tuple[float, float]:

    mileage = train_data["km"]
    price  = train_data["price"]
    m = len(mileage)
    #print(f"M vaut: {m}")
    sum_error_theta0 = 0.0
    sum_error_theta1 = 0.0

    for i in range(m):
        prediction = estimate_price(mileage.iloc[i], theta0, theta1)
        error = prediction - float(price.iloc[i])
        sum_error_theta0 += error
        sum_error_theta1 += error * float(mileage.iloc[i])

    gradient_theta0 = sum_error_theta0 / m
    gradient_theta1 = sum_error_theta1 / m

    theta0 = theta0 - learning_rate*gradient_theta0
    theta1 = theta1 - learning_rate*gradient_theta1

    print(f"theta0: {theta0}")
    print(f"theta1: {theta1}")

    return theta0, theta1