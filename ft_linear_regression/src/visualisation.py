import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from . import train


def main() -> None:
    """Bonus : Afficher la regression linéaire, ainsi que valeurs de data.csv.
    Puis rajouter un programme qui calcule la précision de l'algorihtme"""
    data_file = Path(__file__).resolve().parents[1] / "data" / "data.csv"

    try: 
        data = pd.read_csv(data_file)
    except FileNotFoundError:
        print("Error: data file not found")
        return
    except pd.errors.EmptyDataError:
        print("Error: data file is empty")
        return
    except pd.errors.ParserError:
        print("Error: unable to parse data file")
        return

    train_data = data
    theta0, theta1 = train.train(train_data)

    show_regression(
        train_data,
        theta0,
        theta1
    )

if __name__ == "__main__":
    main()