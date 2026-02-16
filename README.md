# Churn MLOps – Production-Oriented ML System

[![Licence](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python version](https://img.shields.io/badge/python-%3E%3D3.12-slim)](https://www.python.org/downloads/)
[![Dernier commit](https://img.shields.io/github/last-commit/arnaudstdr/churn-mlops/main)](https://github.com/arnaudstdr/churn-mlops/commits/main)
[![Stars](https://img.shields.io/github/stars/arnaudstdr/generate_mail?style=social)](https://github.com/arnaudstdr/generate_mail/stargazers)

## Overview

This project is a **production-oriented MLOps case study** built to demonstrate how a classical machine learning use case can be **designed, deployed, and maintained in real-world conditions**.

The chosen business problem is **customer churn prediction**, a common and concrete use case in SaaS, subscription-based services, and telecom environments.

> **Focus:** delivery, reliability, and maintainability — not leaderboard performance.

---

## What this project demonstrates

- How to approach a **real ML problem as a product**, not as a notebook
- How to structure an ML project with **production constraints in mind**
- How MLOps practices fit into a realistic delivery workflow
- How to progressively industrialize a model (API, deployment, monitoring)

This project reflects how I would approach a **client mission or internal product**, step by step.

---

## Business problem

Customer churn has a direct impact on revenue and growth.
Companies often struggle to:
- identify at-risk users early enough
- operationalize ML predictions
- trust models once deployed in production

Many churn projects fail not because of the model, but because of **poor integration and lack of monitoring**.

---

## Approach (production mindset)

This project deliberately prioritizes:

- Clear project structure
- Explicit assumptions and trade-offs
- Incremental delivery
- Observability and lifecycle management

The goal is **not** to build the most complex pipeline, but a **robust and understandable one**.

---

## Current status

Sprints 0–3 completed. The project includes a trained model, a production API, ML tests, linting, and MLflow tracking. CI/CD (Sprint 4) is next.

### Modeling

A baseline churn prediction model is implemented using logistic regression, trained on the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn).

**Test set metrics (model v1.0.0)**

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.8636 |
| Precision | 0.6820 |
| Recall | 0.6312 |
| F1-score | 0.6556 |

Artefacts: `models/logistic_regression_model.joblib`, `models/preprocessor.joblib`

#### Preprocessing pipeline

- **One-Hot Encoding** for categorical variables (gender, contract type, etc.)
- **Standard Scaling** for numerical variables (tenure, monthly charges)
- **Imputation** — mode for categorical, median for numerical

#### Training

To retrain the model:

```bash
python -m ml.train
# or with a custom dataset:
python -m ml.train --data-path path/to/data.csv
```

Parameters: `max_iter=1000`, `random_state=42`, `threshold=0.5`, splits 70/15/15.

Each run is tracked in MLflow (parameters, metrics, artifacts, git commit).

---

## API

The prediction model is exposed via a REST API built with FastAPI.

### Endpoints

#### `GET /health`

Returns the current status of the API and whether the model is loaded.

**Response `200`**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

---

#### `GET /model`

Returns metadata about the loaded model.

**Response `200`**
```json
{
  "model_type": "LogisticRegression",
  "model_version": "1.0.0",
  "features": ["gender", "SeniorCitizen", "tenure", "..."],
  "training_date": null
}
```

---

#### `POST /predict`

Predicts churn probability for a given customer.

**Request body**
```json
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
  "TotalCharges": "3300.0"
}
```

**Response `200`**
```json
{
  "churn_prediction": false,
  "churn_probability": 0.07,
  "model_version": "1.0.0",
  "request_id": "a1b2c3d4-..."
}
```

**Error codes**

| Code | Cause |
|------|-------|
| `422` | Missing or invalid field |
| `503` | Model not loaded |
| `500` | Internal prediction error |

### Headers

Every response includes a `X-Request-ID` header matching the `request_id` field in the response body. This can be used to correlate logs and Sentry events.

---

## MLflow

MLflow is used to track experiments, parameters, metrics, and artifacts across training runs.

### Start the MLflow UI

```bash
docker compose up mlflow
```

The UI is available at `http://localhost:5000`.

Each training run logs:
- **Parameters**: splits, `max_iter`, `random_state`, `threshold`
- **Metrics**: `val_*` and `test_*` for ROC-AUC, precision, recall, F1
- **Artifacts**: `.joblib` model and preprocessor files, sklearn model
- **Tags**: `git_commit`, `model_type`, `dataset`

---

## Roadmap

| Sprint | Status | Scope |
|--------|--------|-------|
| 0 — Foundation | ✅ Done | Repo, hooks, Sentry |
| 1 — ML offline | ✅ Done | Baseline model, preprocessing, artefacts |
| 2 — API | ✅ Done | FastAPI, contract, tests, observability |
| 3 — Monitoring & quality | ✅ Done | MLflow, ML tests, lint (ruff) |
| 4 — CI & stabilisation | 🔜 Next | GitHub Actions, badges, cleanup |

---

## Philosophy

- Pragmatic, production-first approach
- No over-engineering
- Explicit design choices
- Code meant to be read, maintained, and reused

---

## Git hooks

This repository uses shared Git hooks to enforce basic quality standards.

After cloning:

```bash
./scripts/install-hooks.sh
```

---

## Observability

### Overview

Observability is a core component of this project. It ensures that we can monitor the health, performance, and behavior of our system in production. This includes tracking errors, logging structured data, and monitoring model performance over time.

### Tools

#### Sentry

We use **Sentry** for error tracking and monitoring. Sentry helps us:

- Capture and track errors in real-time.
- Monitor API performance and health.
- Correlate errors with specific requests using `request_id`.
- Set up alerts for critical issues.

### Logging

Structured logging is essential for debugging and monitoring. We use JSON logging to ensure that logs are easily parseable and can be integrated with logging platforms.

## Author
Arnaud Stadler | Python Developer - AI Automation & MLOps (Productive & Delivery)
