"""Tests for the Customer Churn Prediction API."""

import uuid


# ── GET /health ──────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_healthy(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_has_request_id_header(self, client):
        response = client.get("/health")

        request_id = response.headers.get("X-Request-ID")
        assert request_id is not None
        # Vérifie le format UUID
        uuid.UUID(request_id)


# ── GET /model ───────────────────────────────────────────────────────────────


class TestModelEndpoint:
    def test_model_info(self, client):
        response = client.get("/model")

        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "model_version" in data
        assert isinstance(data["features"], list)
        assert len(data["features"]) > 0


# ── POST /predict ────────────────────────────────────────────────────────────


class TestPredictEndpoint:
    def test_predict_low_churn(self, client, sample_customer):
        response = client.post("/predict", json=sample_customer)

        assert response.status_code == 200
        data = response.json()
        assert data["churn_prediction"] is False
        assert 0 <= data["churn_probability"] <= 1
        assert "request_id" in data

    def test_predict_high_churn(self, client, high_churn_customer):
        response = client.post("/predict", json=high_churn_customer)

        assert response.status_code == 200
        data = response.json()
        assert data["churn_prediction"] is True
        assert 0 <= data["churn_probability"] <= 1

    def test_predict_missing_fields(self, client):
        incomplete = {"gender": "Male", "SeniorCitizen": 1}
        response = client.post("/predict", json=incomplete)

        assert response.status_code == 422

    def test_predict_invalid_types(self, client, sample_customer):
        bad_payload = {**sample_customer, "tenure": "abc"}
        response = client.post("/predict", json=bad_payload)

        assert response.status_code == 422

    def test_predict_request_id_header_matches_body(self, client, sample_customer):
        response = client.post("/predict", json=sample_customer)

        assert response.status_code == 200
        header_id = response.headers.get("X-Request-ID")
        body_id = response.json()["request_id"]
        assert header_id == body_id
