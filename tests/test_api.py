#!/usr/bin/env python3
"""
Test script for the Customer Churn Prediction API.
"""

import sys
import os
import requests
import json
from fastapi.testclient import TestClient

# Add the project root to Python path so we can import the api module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app

def test_health_endpoint():
    """Test the health endpoint."""
    client = TestClient(app)
    response = client.get("/health")

    print("Health endpoint test:")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_model_endpoint():
    """Test the model info endpoint."""
    client = TestClient(app)
    response = client.get("/model")

    print("Model endpoint test:")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "model_version" in data
    assert "features" in data

def test_predict_endpoint():
    """Test the predict endpoint with sample data."""
    client = TestClient(app)

    # Sample customer data (from the dataset)
    sample_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85"
    }

    print("Predict endpoint test:")
    print(f"Input data: {json.dumps(sample_data, indent=2)}")

    response = client.post("/predict", json=sample_data)

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "model_version" in data
    assert "request_id" in data

    # Verify that probability is between 0 and 1
    assert 0 <= data["churn_probability"] <= 1

def test_invalid_input():
    """Test the predict endpoint with invalid input."""
    client = TestClient(app)

    # Invalid data (missing required field)
    invalid_data = {
        "gender": "Male",
        "SeniorCitizen": 1,
        # Missing other required fields
    }

    print("Invalid input test:")
    response = client.post("/predict", json=invalid_data)

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

    assert response.status_code == 422  # Unprocessable Entity for validation errors

if __name__ == "__main__":
    print("Running API tests...\n")

    try:
        test_health_endpoint()
        test_model_endpoint()
        test_predict_endpoint()
        test_invalid_input()

        print("✅ All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise