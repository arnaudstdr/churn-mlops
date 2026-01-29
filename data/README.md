# Telco Customer Churn Dataset

## Overview

This dataset contains information about customers of a telecommunications company, including their demographics, services, and churn status. The goal is to predict customer churn based on these features.

## Data Inspection

### Data Types

| Column | Type | Description |
|--------|------|-------------|
| customerID | object | Unique customer identifier |
| gender | object | Customer gender (Male/Female) |
| SeniorCitizen | int64 | Whether the customer is a senior citizen (1/0) |
| Partner | object | Whether the customer has a partner (Yes/No) |
| Dependents | object | Whether the customer has dependents (Yes/No) |
| tenure | int64 | Number of months the customer has been with the company |
| PhoneService | object | Whether the customer has phone service (Yes/No) |
| MultipleLines | object | Whether the customer has multiple lines (Yes/No/No phone service) |
| InternetService | object | Type of internet service (DSL/Fiber optic/No) |
| OnlineSecurity | object | Whether the customer has online security (Yes/No/No internet service) |
| OnlineBackup | object | Whether the customer has online backup (Yes/No/No internet service) |
| DeviceProtection | object | Whether the customer has device protection (Yes/No/No internet service) |
| TechSupport | object | Whether the customer has tech support (Yes/No/No internet service) |
| StreamingTV | object | Whether the customer has streaming TV (Yes/No/No internet service) |
| StreamingMovies | object | Whether the customer has streaming movies (Yes/No/No internet service) |
| Contract | object | Type of contract (Month-to-month/One year/Two year) |
| PaperlessBilling | object | Whether the customer has paperless billing (Yes/No) |
| PaymentMethod | object | Payment method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)) |
| MonthlyCharges | float64 | Monthly charges for the customer |
| TotalCharges | object | Total charges for the customer |
| Churn | object | Whether the customer churned (Yes/No) |

### Missing Values

The dataset does not contain any missing values. All columns are fully populated.

### Unique Values

- **customerID**: 7043 unique values (one per customer)
- **gender**: 2 unique values (Male, Female)
- **SeniorCitizen**: 2 unique values (0, 1)
- **Partner**: 2 unique values (Yes, No)
- **Dependents**: 2 unique values (Yes, No)
- **tenure**: 73 unique values (range: 1 to 72 months)
- **PhoneService**: 2 unique values (Yes, No)
- **MultipleLines**: 3 unique values (Yes, No, No phone service)
- **InternetService**: 3 unique values (DSL, Fiber optic, No)
- **OnlineSecurity**: 3 unique values (Yes, No, No internet service)
- **OnlineBackup**: 3 unique values (Yes, No, No internet service)
- **DeviceProtection**: 3 unique values (Yes, No, No internet service)
- **TechSupport**: 3 unique values (Yes, No, No internet service)
- **StreamingTV**: 3 unique values (Yes, No, No internet service)
- **StreamingMovies**: 3 unique values (Yes, No, No internet service)
- **Contract**: 3 unique values (Month-to-month, One year, Two year)
- **PaperlessBilling**: 2 unique values (Yes, No)
- **PaymentMethod**: 4 unique values (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- **MonthlyCharges**: 1585 unique values (range: 18.25 to 118.75)
- **TotalCharges**: 6531 unique values (range: 18.8 to 8684.8)
- **Churn**: 2 unique values (Yes, No)

### Hypotheses

1. **SeniorCitizen**: Senior citizens are more likely to churn due to potential difficulties with technology or changing needs.
2. **Partner and Dependents**: Customers with partners or dependents are less likely to churn as they may have more stable living situations.
3. **tenure**: Customers with longer tenure are less likely to churn as they are more established with the service.
4. **Contract**: Customers with month-to-month contracts are more likely to churn compared to those with longer-term contracts.
5. **MonthlyCharges**: Higher monthly charges are associated with a higher likelihood of churn.
6. **TotalCharges**: Higher total charges are associated with a lower likelihood of churn, as these customers have invested more in the service.
7. **InternetService**: Customers with fiber optic internet service are more likely to churn due to potential higher costs or expectations.
8. **TechSupport**: Customers without tech support are more likely to churn due to potential frustration with unresolved issues.
9. **PaymentMethod**: Customers using electronic checks are more likely to churn compared to those using automatic payment methods.

### Data Quality Issues

- **TotalCharges**: This column is stored as an object (string) instead of a numeric type. It should be converted to a numeric type for analysis.
- **SeniorCitizen**: This column is stored as an integer (1/0) instead of a categorical type (Yes/No). It should be converted to a categorical type for consistency.

### Next Steps

1. Convert `TotalCharges` to a numeric type.
2. Convert `SeniorCitizen` to a categorical type (Yes/No).
3. Encode categorical variables for modeling.
4. Split the data into training and testing sets.
5. Train a baseline model to predict churn.