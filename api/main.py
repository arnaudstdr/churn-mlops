from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sentry_sdk
import uuid
import logging
from typing import Optional

from .schemas import CustomerFeatures, PredictionResult, HealthCheck, ModelInfo
from .service import ChurnPredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

sentry_dsn = os.getenv("SENTRY_DSN")

if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        # Add data like request headers and IP for users,
        # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
        send_default_pii=True,
    )

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0"
)

# Initialize prediction service
prediction_service = ChurnPredictionService()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint to verify API status.

    Returns:
        HealthCheck: Status information including model loading status
    """
    try:
        model_ready = prediction_service.is_ready()

        if model_ready:
            model_info = prediction_service.get_model_info()
            return HealthCheck(
                status="healthy",
                model_loaded=True,
                model_version=model_info.get("model_version", "unknown")
            )
        else:
            return HealthCheck(
                status="degraded",
                model_loaded=False,
                model_version=None
            )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheck(
            status="unhealthy",
            model_loaded=False,
            model_version=None
        )


@app.get("/model", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the loaded model.

    Returns:
        ModelInfo: Detailed information about the model
    """
    try:
        if not prediction_service.is_ready():
            raise HTTPException(status_code=503, detail="Model not loaded")

        model_info = prediction_service.get_model_info()

        return ModelInfo(
            model_type=model_info["model_type"],
            model_version=model_info["model_version"],
            features=model_info["features"],
            training_date=None  # Could be added if available
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@app.post("/predict", response_model=PredictionResult)
async def predict_churn(customer_data: CustomerFeatures):
    """
    Predict customer churn based on input features.

    Args:
        customer_data: CustomerFeatures containing all required customer information

    Returns:
        PredictionResult: Prediction result with probability and metadata

    Raises:
        HTTPException: 400 if input validation fails or prediction error occurs
        HTTPException: 503 if model is not loaded
    """
    try:
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())
        logger.info(f"Received prediction request {request_id}")

        # Check if service is ready
        if not prediction_service.is_ready():
            logger.error(f"Request {request_id}: Model not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert customer data to dict and make prediction
        input_data = customer_data.model_dump()
        churn_prediction, churn_probability = prediction_service.predict(input_data)

        logger.info(f"Request {request_id}: Prediction completed - churn_probability={churn_probability:.4f}")

        return PredictionResult(
            churn_prediction=churn_prediction,
            churn_probability=churn_probability,
            model_version=prediction_service.model_version,
            request_id=request_id
        )

    except ValueError as ve:
        logger.error(f"Request {request_id}: Validation error - {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        logger.error(f"Request {request_id}: Prediction error - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")