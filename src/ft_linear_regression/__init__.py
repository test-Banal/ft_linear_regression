import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import train
import utils_eda


def compare_predictions(
    data: pd.DataFrame,
    theta0: float,
    theta1: float,
) -> None:
    max_mileage = float(data["km"].max())

    mileage = data["km"]
    real_prices = data["price"]

    normalized_mileage = mileage / max_mileage

    predicted_prices = (
        theta0 + theta1 * normalized_mileage
    )
    for i in range(len(data)):
        print(
            f"{mileage.iloc[i]:.0f} km | "
            f"Real: {real_prices.iloc[i]:.2f} € | "
            f"Predicted: {predicted_prices.iloc[i]:.2f} €"
        )
    
    plt.scatter(
        mileage,
        real_prices,
        color="blue",
        label="Real prices",
    )

    plt.scatter(
        mileage,
        predicted_prices,
        color="red",
        label="Predicted prices",
    )

    plt.plot(
        mileage,
        predicted_prices,
        label="Linear regression",
    )

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Real prices vs predicted prices")
    plt.legend()
    plt.show()


def main() -> None:
    # resolve() supprime les segments relatifs comme "../.." du chemin.
    data_file = Path(__file__).resolve().parents[2] / "data" / "data.csv"
    data = pd.read_csv(data_file)
    train_data = data
    #utils_eda.test_info(data)
    #utils_eda.test_show(data)
    #utils_edaprint_fin(data)
    #train.training_try(train_data)

    theta0, theta1 = train.train(train_data)

    """compare_predictions(
        train_data,
        theta0,
        theta1,
    )"""   

    print( 
        f"theta0={theta0:.2f}, "
        f"theta1={theta1:.2f}"
    )
    max_mileage = float(train_data["km"].max())
    train.save_parameters(
        theta0,
        theta1,
        max_mileage,
)

if __name__ == "__main__":
    main()
