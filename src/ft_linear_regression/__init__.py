import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import train
import utils_eda
import visualisation

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
    
    #utils_eda.test_info(data)
    #utils_eda.test_show(data)
    #utils_edaprint_fin(data)
    #train.training_try(train_data)
if __name__ == "__main__":
    main()
