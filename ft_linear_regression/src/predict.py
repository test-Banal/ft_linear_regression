import json
from pathlib import Path
import pandas as pd

def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    """Calcule le prix estimé de la voiture en fonction de son kilométrage"""
    return theta0 + theta1 * mileage



def load_parameters() -> tuple[float, float, float]:
    """Lit le JSON du modèle, et renvoie les valeurs de theta0, theta1, et le kilométrage"""
    model_file = (
            Path(__file__).resolve().parents[1]
            / "model" / "parameters.json"
            )

    with model_file.open("r") as file:
        parameters = json.load(file)
    
    theta0 = float(parameters["theta0"])
    theta1 = float(parameters["theta1"])
    max_mileage = float(parameters["max_mileage"])
    return theta0, theta1, max_mileage


def main() -> None:
    "Objectif : Valider les données, et la saisie (mileage), puis affiche resultat"
    root = Path(__file__).resolve().parents[1] #resolve = chemin relatif a absolu; en remonté de deux niveaux dans l'arborescence
    model_file = root / "model" / "parameters.json" #construit les chemin absolu aux fichiers pour predict.py
    data_file = root / "data" / "data.csv" 

    if not model_file.exists():
        print("0")
        print("Error: model has not been trained yet.")
        return
    try:
        theta0, theta1, max_mileage = load_parameters()
    except json.JSONDecodeError:#texte qui ne respecte pas la syntaxe JSON
        print("0")
        print("Error: model parameters file is corrupted.")
        return
    except KeyError:
        print("0")
        print("Error: model parameters are incomplete.")
        return
    if max_mileage <= 0:
        print("Error: invalid model value.")

    try:
        data = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: data file not fo() und: {data_file}")
        return
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        print(f"Error: invalid data file: {data_file}")
        return

    if data.empty or not {"km", "price"}.issubset(data.columns): #issubset = verifie si ts les elements d'un ensemble sont contenus dans un autre ensemble
        print("Error: dataset must contain non-empty 'km' and 'price' columns")
        return
    if data[["km", "price"]].isna().any().any():
        print("Error: dataset contains missing values.")
        return

    try: 
        mileage = float(input("Mileage: "))
    except ValueError:
        print("Error: mileage must be a number.")
        return
    if mileage < 0:
        print("Error: mileage cannot be negative.")
        return

    normalized_mileage = mileage / max_mileage

    predicted_price = estimate_price(
            normalized_mileage,
            theta0,
            theta1
    )

    print(f"Estimated price: {predicted_price:.2f} €")

if __name__ == "__main__":
    main()
