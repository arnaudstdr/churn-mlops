"""Tests ML : preprocessing, smoke predict, non-régression."""

import numpy as np
import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_input():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.0,
        "TotalCharges": "3300.0",
    }


# ── Preprocessing ─────────────────────────────────────────────────────────────


class TestPreprocessing:
    def test_output_is_2d(self, service, sample_input):
        result = service.preprocess_input(sample_input)
        assert result.ndim == 2

    def test_output_has_one_row(self, service, sample_input):
        result = service.preprocess_input(sample_input)
        assert result.shape[0] == 1

    def test_output_has_no_nan(self, service, sample_input):
        result = service.preprocess_input(sample_input)
        assert not np.isnan(result.toarray() if hasattr(result, "toarray") else result).any()

    def test_output_is_numeric(self, service, sample_input):
        result = service.preprocess_input(sample_input)
        arr = result.toarray() if hasattr(result, "toarray") else result
        assert arr.dtype.kind in ("f", "i", "u")

    def test_missing_field_raises(self, service, sample_input):
        incomplete = {k: v for k, v in sample_input.items() if k != "tenure"}
        with pytest.raises(Exception):
            service.preprocess_input(incomplete)


# ── Smoke predict ─────────────────────────────────────────────────────────────


class TestPredictSmoke:
    def test_returns_tuple(self, service, sample_input):
        result = service.predict(sample_input)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_prediction_is_bool(self, service, sample_input):
        churn_prediction, _ = service.predict(sample_input)
        assert isinstance(churn_prediction, bool)

    def test_probability_in_range(self, service, sample_input):
        _, churn_probability = service.predict(sample_input)
        assert 0.0 <= churn_probability <= 1.0

    def test_custom_threshold(self, service, sample_input):
        _, proba = service.predict(sample_input)
        pred_low, _ = service.predict(sample_input, threshold=0.0)
        pred_high, _ = service.predict(sample_input, threshold=1.0)
        assert pred_low is True
        assert pred_high is False


# ── Non-régression ────────────────────────────────────────────────────────────
#
# Ces cas sont fixés à partir des sorties observées du modèle v1.0.0.
# Un échec ici signifie que le modèle ou le preprocessing a changé.


NON_REGRESSION_CASES = [
    # (input, expected_prediction, expected_proba_approx)
    (
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 55.0,
            "TotalCharges": "3300.0",
        },
        False,
        0.0062,  # proba attendue ± tolérance
    ),
    (
        {
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
            "TotalCharges": "95.0",
        },
        True,
        0.8651,
    ),
]


class TestNonRegression:
    @pytest.mark.parametrize("input_data,expected_pred,expected_proba", NON_REGRESSION_CASES)
    def test_prediction_stable(self, service, input_data, expected_pred, expected_proba):
        pred, proba = service.predict(input_data)
        assert pred == expected_pred, f"Prediction changed: got {pred}, expected {expected_pred}"
        assert abs(proba - expected_proba) < 0.01, (
            f"Probability drifted: got {proba:.4f}, expected {expected_proba:.4f} (±0.01)"
        )
