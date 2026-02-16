"""
Script d'entraînement du modèle de churn avec tracking MLflow.

Usage:
    python -m ml.train
    python -m ml.train --data-path data/Telco-Customer-Churn.csv
"""

import argparse
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Chemins par défaut ────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"

# ── Paramètres d'entraînement ─────────────────────────────────────────────────

PARAMS = {
    "test_size": 0.15,
    "val_size": 0.15,
    "random_state": 42,
    "max_iter": 1000,
    "threshold": 0.5,
}


# ── Fonctions ─────────────────────────────────────────────────────────────────


def load_data(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numerical_transformer, numerical_cols),
        ]
    )


def compute_metrics(model, X, y) -> dict:
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= PARAMS["threshold"]).astype(int)
    return {
        "roc_auc": roc_auc_score(y, y_proba),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


# ── Pipeline principal ────────────────────────────────────────────────────────


def train(data_path: Path = DATA_PATH) -> None:
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        # Tags de traçabilité
        mlflow.set_tags(
            {
                "git_commit": get_git_commit(),
                "model_type": "LogisticRegression",
                "dataset": data_path.name,
            }
        )

        # Paramètres
        mlflow.log_params(PARAMS)

        # Chargement et préparation des données
        data = load_data(data_path)
        X_raw = data.drop(columns=["customerID", "Churn"])
        y = data["Churn"].map({"Yes": 1, "No": 0})

        preprocessor = build_preprocessor(X_raw)
        X = preprocessor.fit_transform(X_raw)

        # Split train / val / test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=PARAMS["test_size"], random_state=PARAMS["random_state"]
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=PARAMS["val_size"], random_state=PARAMS["random_state"]
        )

        mlflow.log_params(
            {
                "n_train": X_train.shape[0],
                "n_val": X_val.shape[0],
                "n_test": X_test.shape[0],
                "n_features": X_train.shape[1],
            }
        )

        # Entraînement
        model = LogisticRegression(
            max_iter=PARAMS["max_iter"], random_state=PARAMS["random_state"]
        )
        model.fit(X_train, y_train)

        # Métriques
        val_metrics = compute_metrics(model, X_val, y_val)
        test_metrics = compute_metrics(model, X_test, y_test)

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        print("\nValidation:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nTest:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Sauvegarde des artefacts
        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / "logistic_regression_model.joblib"
        preprocessor_path = MODEL_DIR / "preprocessor.joblib"

        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(preprocessor_path))
        mlflow.sklearn.log_model(model, "model")

        print(f"\nArtefacts sauvegardés dans {MODEL_DIR}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    train(data_path=args.data_path)
