"""Fournit les outils d'analyse exploratoire des données de voitures.

Les fonctions permettent de contrôler la qualité des données, de consulter
leurs statistiques et de visualiser le lien entre kilométrage et prix."""

import pandas as pd
import matplotlib.pyplot as plt



def test_info(data: pd.DataFrame) -> None:
    """Affiche les informations utiles pour l'EDA et le contrôle des données."""
    print(data)  # Utiliser data.head() pour n'afficher que les premières lignes.
    print(data.shape)
    print()
    print(data.columns)
    data.info()
    print()
    print("Valeurs manquantes")
    print(data.isna().sum())
    print("Doublons :")
    print(data.duplicated().sum())
    print("Valeurs km anormales")
    print(data[data["km"] < 0])
    print("Valeurs prix anormales")
    print(data[data["price"] < 0])
    print("Informations supplémentaires")
    print(data.describe())
    print()


def test_show(data: pd.DataFrame) -> None:
    """Affiche le graphique du prix en fonction du kilométrage."""
    plt.scatter(data["km"], data["price"])
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Price according to mileage")
    plt.show()


def print_fin(data: pd.DataFrame) -> None:
    """Affiche la corrélation et les données triées par kilométrage."""
    print("Corrélation :")
    print(data.corr(numeric_only=True))
    print()
    print("Données triées par kilométrage :")
    df_sorted = data.sort_values(by="km")
    print(df_sorted)



'''import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import train
import utilseda
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
'''