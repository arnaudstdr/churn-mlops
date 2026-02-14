from pydantic import BaseModel
from typing import Optional


class CustomerFeatures(BaseModel):
    """
    Input features for customer churn prediction.

    This schema represents all the features needed to predict customer churn,
    excluding the customerID and the target variable (Churn).
    """
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str


class PredictionResult(BaseModel):
    """
    Output schema for churn prediction results.

    Contains the prediction (churn or not), the probability score,
    and the version of the model used.
    """
    churn_prediction: bool
    churn_probability: float
    model_version: str
    request_id: Optional[str] = None


class HealthCheck(BaseModel):
    """
    Response schema for health check endpoint.
    """
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ModelInfo(BaseModel):
    """
    Response schema for model information endpoint.
    """
    model_type: str
    model_version: str
    features: list[str]
    training_date: Optional[str] = None