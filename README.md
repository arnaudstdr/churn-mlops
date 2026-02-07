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

🚧 **Early-stage project**

At this stage, the project focuses on:
- defining the scope
- setting up the project structure
- establishing quality and workflow standards

Technical components (modeling, API, CI/CD, monitoring, deployment) will be added incrementally and documented as the project evolves.

### Modeling

A baseline churn prediction model has been implemented using logistic regression. The model is trained on the Telco Customer Churn dataset and achieves the following performance metrics:

- **ROC-AUC**: 0.8636
- **Precision**: 0.6820
- **Recall**: 0.6312
- **F1-score**: 0.6556

The model is saved as `models/logistic_regression_model.joblib` and the preprocessing pipeline is saved as `models/preprocessor.joblib`.

#### Data Preprocessing

The preprocessing pipeline includes:
- **One-Hot Encoding**: For categorical variables (e.g., gender, contract type).
- **Standard Scaling**: For numerical variables (e.g., tenure, monthly charges).
- **Imputation**: Handling missing values by replacing them with the mode (categorical) or median (numerical).

#### Model Training

The logistic regression model is trained with the following parameters:
- `max_iter`: 1000
- `random_state`: 42

The model is evaluated on both validation and test sets to ensure robustness and generalization.

---

## Planned evolution (high-level roadmap)

Planned steps include:
- baseline churn model
- training and evaluation pipeline
- API exposure of predictions
- containerized deployment
- monitoring of predictions and data drift
- CI/CD automation

Each step will be implemented with production constraints in mind.

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
