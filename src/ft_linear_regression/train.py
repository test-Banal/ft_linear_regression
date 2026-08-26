import json
import pandas as pd
from pathlib import Path
import predict

def save_parameters(theta0: float, theta1: float, max_mileage: float) -> None:
    """Sauvegarde valeurs de theta0 et theta1 dans fichier JSON pour ne pas recommencer les calculs : bonne pratique
    """
    model_file = (
        Path(__file__).resolve().parents[2]
        / "model"
        / "parameters.json"
    )

    parameters = {
        "theta0": theta0,
        "theta1": theta1,
        "max_mileage": max_mileage,
    }

    with model_file.open("w") as file:
        json.dump(parameters, file, indent=4)


def training(train_data: pd.DataFrame) -> tuple[float, float]:
    theta0 = 0.0
    theta1 = 0.0
    learning_rate = 0.1
    iterations = 5000 #essayer avec 1000, 2000 et on voit que ca ne bouge plus avec un grand nombre d'iterations
    for i in range(iterations):
        theta0, theta1 = train_step(
            train_data,
            theta0,
            theta1,
            learning_rate,
        )

        if i % 100 == 0 or i == iterations - 1:
            cost = cost_function(
                train_data,
                theta0,
                theta1,
            )
            print(
                f"Iteration {i}: "
                f"cost={cost:.2f}, "
                f"theta0={theta0:.2f}, "
                f"theta1={theta1:.2f}"
            )
    return (theta0, theta1)


def training_try(train_data: pd.DataFrame) -> None:
    mileage = train_data["km"]
    price = train_data["price"]

    theta0 = 0.0
    theta1 = 0.0

    prediction = predict.estimate_price(
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

def train_step(train_data: pd.DataFrame, theta0: float, theta1: float, learning_rate:float) -> tuple[float, float]:
    """effectuer une etape de gradient descent
    """
#    mileage = train_data["km"]
    mileage = train_data["km"] / train_data["km"].max()
    price  = train_data["price"]
    m = len(mileage)
    #print(f"M vaut: {m}")
    sum_error_theta0 = 0.0
    sum_error_theta1 = 0.0

    for i in range(m):
        prediction = predict.estimate_price(mileage.iloc[i], theta0, theta1)
        error = prediction - float(price.iloc[i])
        sum_error_theta0 += error
        sum_error_theta1 += error * float(mileage.iloc[i])

    gradient_theta0 = sum_error_theta0 / m
    gradient_theta1 = sum_error_theta1 / m

    theta0 = theta0 - learning_rate*gradient_theta0
    theta1 = theta1 - learning_rate*gradient_theta1



    return theta0, theta1

def cost_function(train_data: pd.DataFrame, theta0: float, theta1: float,) -> float:
    """Mesure l'erreur
    """
    mileage = train_data["km"] / train_data["km"].max()
    price = train_data["price"]

    m = len(mileage)
    total_error = 0.0

    for i in range(m):
        prediction = predict.estimate_price(
            float(mileage.iloc[i]),
            theta0,
            theta1,
        )

        error = prediction - float(price.iloc[i])

        total_error += error ** 2

    return total_error / (2 * m)