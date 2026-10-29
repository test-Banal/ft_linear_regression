import json
import math
from pathlib import Path
import pandas as pd
from predict import estimate_price


def save_parameters(theta0: float, theta1: float, max_mileage: float) -> None:
    """Sauvegarde valeurs dans fichier JSON"""
    model_file = Path(__file__).resolve().parents[1] / "model" / "parameters.json"

    parameters = {"theta0": theta0,
                  "theta1": theta1,
                  "max_mileage":max_mileage
                  }

    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("w") as file:
        json.dump(parameters, file, indent=4)

def train(train_data: pd.DataFrame) -> tuple[float, float]:
    """Effectue 5000 etapes de descente de gradient, et renvoie les parametres obtenus"""
    theta0 = 0.0
    theta1 = 0.0
    learning_rate = 0.1 #La taille du pas a chaque MAJ de theta0 et theta1. Le gradient indique dans quelle direction les paramètres font augmenter l’erreur. On le soustrait pour réduire l’erreur, et learning_rate règle l’amplitude de cette correction.
    iterations = 5000
    for i in range(iterations):
        theta0, theta1 = train_step(
                train_data, 
                theta0,
                theta1,
                learning_rate
                )
    return (theta0, theta1)

def train_step(train_data: pd.DataFrame, theta0: float, theta1: float, learning_rate: float) -> tuple[float, float]:
    """Normalise les kilometrages, mets a jours les deux parametres avec la descente de gradient"""
    mileage = train_data["km"] / train_data["km"].max()
    price = train_data["price"]
    m = len(mileage) #m = index du nombre de kilometre, ici 24 lignes, ensuite m pour : 1) parcourir toutes les voitures avec for i in range(m)
                     #                                                                  2) calculer la moyenne des contributions aux gradients.
                     # diviser par m : permet de faire une MAJ fondée sur l'erreur moyenne, plutot que sur une somme qui grossirait avec le nombre de voiture
    
    sum_error_theta0 = 0.0
    sum_error_theta1 = 0.0

    for i in range(m):
        prediction = estimate_price(float(mileage.iloc[i]), theta0, theta1)
        error = prediction - float(price.iloc[i])
        sum_error_theta0 += error
        sum_error_theta1 += error * float(mileage.iloc[i])

    gradient_theta0 = sum_error_theta0 / m
    gradient_theta1 = sum_error_theta1 / m

    theta0 = theta0 - learning_rate*gradient_theta0
    theta1 = theta1 - learning_rate*gradient_theta1

    return theta0, theta1


def main() -> None:
    """Charge et contrôle les données, entraîne le modèle et sauvegarde ses paramètres."""
    data_file = Path(__file__).resolve().parents[1] / "data" / "data.csv"
    data = pd.read_csv(data_file)
    train_data = data
    required_columns = {"km", "price"}

    if not required_columns.issubset(train_data.columns):
        print("Error: dataset must contain 'km' and 'price' columns.")
        return

    if train_data.empty:
        print("Error: dataset is empty.")
        return

    if train_data[["km", "price"]].isna().any().any():
        print("Error: dataset contains missing values.")
        return

    if (train_data["km"] < 0).any():
        print("Error: mileage values cannot be negative.")
        return

    if (train_data["price"] < 0).any():
        print("Error: price values cannot be negative.")
        return

    max_mileage = float(train_data["km"].max())
    if max_mileage <= 0:
        print("Error: dataset must contain a positive mileage.")
        return
    theta0, theta1 = train(train_data)

    #print(
    #        f"theta0={theta0:.2f}, 
    #        f"theta1={theta1:.2f}
    #)

    max_mileage = float(max_mileage)  #conserve facteur de normalisation utilisé pendant l'entrainement pour l'appliquer exactement le meme calcul dans predict.py
    save_parameters(theta0, theta1, max_mileage)
    print(f"Training complete: theta0={theta0:.2f}, theta1={theta1:.2f}")

if __name__ == "__main__":
    main()
