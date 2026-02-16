from pathlib import Path
from typing import Union

import pandas as pd


def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Charge les données depuis un fichier CSV.

    Args:
        data_path: Chemin vers le fichier CSV.

    Returns:
        DataFrame contenant les données.
    """
    return pd.read_csv(data_path)

def inspect_data(df: pd.DataFrame) -> None:
    """Inspecte les données et affiche des statistiques.

    Args:
        df: DataFrame contenant les données.
    """
    print("=== Aperçu des données ===")
    print(df.head())

    print("\n=== Informations sur les données ===")
    print(df.info())

    print("\n=== Statistiques descriptives ===")
    print(df.describe(include="all"))

    print("\n=== Valeurs manquantes ===")
    print(df.isnull().sum())

    print("\n=== Valeurs uniques par colonne ===")
    for column in df.columns:
        print(f"{column}: {df[column].nunique()} valeurs uniques")

def document_hypotheses(df: pd.DataFrame) -> None:
    """Documente les hypothèses principales sur les données.

    Args:
        df: DataFrame contenant les données.
    """
    hypotheses = {
        "customerID": "Identifiant unique du client (non utilisé pour la modélisation).",
        "gender": "Genre du client (Male/Female). Hypothèse : pas de biais de genre dans le churn.",
        "SeniorCitizen": "Indique si le client est un senior (1/0). Hypothèse : les seniors ont un taux de churn plus élevé.",
        "Partner": "Indique si le client a un partenaire (Yes/No). Hypothèse : les clients avec un partenaire ont un taux de churn plus faible.",
        "Dependents": "Indique si le client a des dépendants (Yes/No). Hypothèse : les clients avec des dépendants ont un taux de churn plus faible.",
        "tenure": "Nombre de mois depuis l'inscription du client. Hypothèse : les clients avec une ancienneté plus longue ont un taux de churn plus faible.",
        "PhoneService": "Indique si le client a un service téléphonique (Yes/No). Hypothèse : pas d'impact significatif sur le churn.",
        "MultipleLines": "Indique si le client a plusieurs lignes (Yes/No/No phone service). Hypothèse : les clients avec plusieurs lignes ont un taux de churn plus faible.",
        "InternetService": "Type de service Internet (DSL/Fiber optic/No). Hypothèse : les clients avec un service Internet ont un taux de churn plus élevé.",
        "OnlineSecurity": "Indique si le client a une sécurité en ligne (Yes/No/No internet service). Hypothèse : les clients avec une sécurité en ligne ont un taux de churn plus faible.",
        "OnlineBackup": "Indique si le client a une sauvegarde en ligne (Yes/No/No internet service). Hypothèse : les clients avec une sauvegarde en ligne ont un taux de churn plus faible.",
        "DeviceProtection": "Indique si le client a une protection d'appareil (Yes/No/No internet service). Hypothèse : les clients avec une protection d'appareil ont un taux de churn plus faible.",
        "TechSupport": "Indique si le client a un support technique (Yes/No/No internet service). Hypothèse : les clients avec un support technique ont un taux de churn plus faible.",
        "StreamingTV": "Indique si le client a un streaming TV (Yes/No/No internet service). Hypothèse : les clients avec un streaming TV ont un taux de churn plus élevé.",
        "StreamingMovies": "Indique si le client a un streaming de films (Yes/No/No internet service). Hypothèse : les clients avec un streaming de films ont un taux de churn plus élevé.",
        "Contract": "Type de contrat (Month-to-month/One year/Two year). Hypothèse : les clients avec un contrat mensuel ont un taux de churn plus élevé.",
        "PaperlessBilling": "Indique si le client a une facturation sans papier (Yes/No). Hypothèse : les clients avec une facturation sans papier ont un taux de churn plus élevé.",
        "PaymentMethod": "Méthode de paiement (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)). Hypothèse : les clients avec un paiement automatique ont un taux de churn plus faible.",
        "MonthlyCharges": "Montant des charges mensuelles. Hypothèse : les clients avec des charges mensuelles plus élevées ont un taux de churn plus élevé.",
        "TotalCharges": "Montant total des charges. Hypothèse : les clients avec des charges totales plus élevées ont un taux de churn plus faible.",
        "Churn": "Indique si le client a quitté le service (Yes/No). Variable cible.",
    }

    print("\n=== Hypothèses principales ===")
    for column, hypothesis in hypotheses.items():
        print(f"{column}: {hypothesis}")

def main():
    data_path = Path(__file__).parent.parent / "Telco-Customer-Churn.csv"
    df = load_data(data_path)
    inspect_data(df)
    document_hypotheses(df)

if __name__ == "__main__":
    main()
