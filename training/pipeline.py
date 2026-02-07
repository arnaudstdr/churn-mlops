# training/pipeline.py
"""
Module for training and evaluating a baseline logistic regression model.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from features import load_data, preprocess_data, split_data

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


def evaluate_model(model, X_val, y_val) -> dict:
    """
    Evaluate the model's performance on the validation set.

    Args:
        model: Trained model.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        dict: Dictionary containing performance metrics.
    """
    # Predict probabilities and classes
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    # Calculate metrics
    metrics = {
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "confusion_matrix": confusion_matrix(y_val, y_pred)
    }

    return metrics


def main():
    """
    Main function to train and evaluate the model.
    """
    # Load data
    data = load_data("data/Telco-Customer-Churn.csv")

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model on the validation set
    val_metrics = evaluate_model(model, X_val, y_val)
    
    # Evaluate the model on the test set
    test_metrics = evaluate_model(model, X_test, y_test)
    
    # Display metrics for the validation set
    print("Performance metrics on the validation set:")
    print(f"ROC-AUC : {val_metrics['roc_auc']:.4f}")
    print(f"Precision : {val_metrics['precision']:.4f}")
    print(f"Recall : {val_metrics['recall']:.4f}")
    print(f"F1-score : {val_metrics['f1']:.4f}")
    print(f"Confusion matrix :\n{val_metrics['confusion_matrix']}")
    
    # Display metrics for the test set
    print("\nPerformance metrics on the test set:")
    print(f"ROC-AUC : {test_metrics['roc_auc']:.4f}")
    print(f"Precision : {test_metrics['precision']:.4f}")
    print(f"Recall : {test_metrics['recall']:.4f}")
    print(f"F1-score : {test_metrics['f1']:.4f}")
    print(f"Confusion matrix :\n{test_metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()