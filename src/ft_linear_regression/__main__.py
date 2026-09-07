"""from pathlib import Path

import pandas as pd

from . import train, visualisation


def main() -> None:
    data_file = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "data.csv"
    )

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

    train_data = data

    theta0, theta1 = train.train(train_data)

    visualisation.compare_predictions(
        train_data,
        theta0,
        theta1,
    )


if __name__ == "__main__":
    main()
"""