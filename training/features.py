# training/features.py
"""
Module for preprocessing data from the Telco-Customer-Churn dataset.
"""

import pandas as pd
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
        tuple: (X, y) where X is the DataFrame of preprocessed features and y is the target series.
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

    return X_processed, y


def split_data(X, y, test_size: float = 0.15, val_size: float = 0.15) -> tuple:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Preprocessed features.
        y: Target.
        test_size (float): Size of the test set (default 0.15).
        val_size (float): Size of the validation set (default 0.15).

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """
    Main function to test preprocessing.
    """
    # Load data
    data = load_data("data/Telco-Customer-Churn.csv")

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print(f"Size of the training set: {X_train.shape}")
    print(f"Size of the validation set: {X_val.shape}")
    print(f"Size of the test set: {X_test.shape}")


if __name__ == "__main__":
    main()