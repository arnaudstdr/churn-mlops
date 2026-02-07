# training/export.py
"""
Module for saving model and preprocessing artifacts.
"""

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    data = pd.read_csv(filepath)
    return data


def preprocess_data(data: pd.DataFrame) -> tuple:
    """
    Preprocess data by applying encoding, scaling, and handling missing values.

    Args:
        data (pd.DataFrame): DataFrame containing raw data.

    Returns:
        tuple: (X, y, preprocessor) where X is the DataFrame of preprocessed features, y is the target series, and preprocessor is the preprocessing object.
    """
    # Separate features and target
    X = data.drop(columns=["customerID", "Churn"])
    y = data["Churn"].map({"Yes": 1, "No": 0})  # Convert target to binary (1 for churn, 0 for no churn)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Create a transformer for categorical columns
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Replace missing values with the mode
            ("onehot", OneHotEncoder(handle_unknown="ignore"))  # One-hot encoding
        ]
    )

    # Create a transformer for numerical columns
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # Replace missing values with the median
            ("scaler", StandardScaler())  # Standardization
        ]
    )

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numerical_transformer, numerical_cols)
        ]
    )

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


def train_model(X_train, y_train) -> LogisticRegression:
    """
    Train a logistic regression model.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        LogisticRegression: Trained model.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def save_artefacts(model, preprocessor, model_path: str, preprocessor_path: str) -> None:
    """
    Save the model and preprocessing to files.

    Args:
        model: Trained model.
        preprocessor: Preprocessing object.
        model_path (str): Path to save the model.
        preprocessor_path (str): Path to save the preprocessing.
    """
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Model saved to {model_path}")
    print(f"Preprocessing saved to {preprocessor_path}")


def main():
    """
    Main function to train the model and save artifacts.
    """
    # Load data
    data = load_data("data/Telco-Customer-Churn.csv")

    # Preprocess data
    X, y, preprocessor = preprocess_data(data)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train the model
    model = train_model(X_train, y_train)

    # Save artifacts
    save_artefacts(
        model,
        preprocessor,
        "models/logistic_regression_model.joblib",
        "models/preprocessor.joblib"
    )


if __name__ == "__main__":
    main()