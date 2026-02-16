import json
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any, Dict

import sentry_sdk
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .schemas import CustomerFeatures, HealthCheck, ModelInfo, PredictionResult
from .service import ChurnPredictionService


# Configure JSON logging
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        for field in ("request_id", "endpoint", "method", "status_code", "error", "error_type", "error_details"):
            if field in record.__dict__:
                log_data[field] = record.__dict__[field]

        return json.dumps(log_data, ensure_ascii=False)

# Configure structured JSON logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
json_handler = logging.StreamHandler()
json_handler.setFormatter(JSONFormatter())
logger.addHandler(json_handler)
logger.propagate = False

load_dotenv()

sentry_dsn = os.getenv("SENTRY_DSN")

# Configure Sentry with enhanced settings
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        send_default_pii=True,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.2")),
    )

# Initialize FastAPI app
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

# Middleware for request logging and Sentry context
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware for adding request context to logs and Sentry.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Add request context to logging
    extra = {
        "request_id": request_id,
        "endpoint": request.url.path,
        "method": request.method,
    }

    # Add context to Sentry
    if sentry_dsn:
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("request_id", request_id)
            scope.set_tag("endpoint", request.url.path)
            scope.set_tag("method", request.method)
            scope.set_extra("user_agent", str(request.headers.get("user-agent", "")))

            # Set transaction name for performance monitoring
            transaction_name = f"{request.method} {request.url.path}"
            scope.transaction = transaction_name

            # Add request context
            scope.set_context("request", {
                "url": str(request.url),
                "method": request.method,
                "headers": dict(request.headers),
                "query_params": dict(request.query_params),
            })

    # Process request
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        # Log successful request
        logger.info(
            "Request completed",
            extra={
                **extra,
                "status_code": response.status_code,
                "request_id": request_id
            }
        )

        return response

    except Exception as e:
        # Log error
        logger.error(
            f"Request failed: {str(e)}",
            extra={
                **extra,
                "status_code": 500,
                "error": str(e)
            }
        )

        # Re-raise the exception
        raise


@app.get("/health", response_model=HealthCheck)
async def health_check(request: Request):
    """
    Health check endpoint to verify API status.

    Returns:
        HealthCheck: Status information including model loading status
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    try:
        model_ready = prediction_service.is_ready()

        # Add Sentry context
        if sentry_dsn:
            with sentry_sdk.new_scope() as scope:
                scope.set_tag("endpoint", "/health")
                scope.set_tag("check_type", "health")

        if model_ready:
            model_info = prediction_service.get_model_info()
            logger.info(
                "Health check successful",
                extra={
                    "request_id": request_id,
                    "endpoint": "/health",
                    "status_code": 200,
                    "model_loaded": True
                }
            )
            return HealthCheck(
                status="healthy",
                model_loaded=True,
                model_version=model_info.get("model_version", "unknown")
            )
        else:
            logger.warning(
                "Health check degraded - model not loaded",
                extra={
                    "request_id": request_id,
                    "endpoint": "/health",
                    "status_code": 200,
                    "model_loaded": False
                }
            )
            return HealthCheck(
                status="degraded",
                model_loaded=False,
                model_version=None
            )
    except Exception as e:
        logger.error(
            f"Health check failed: {str(e)}",
            extra={
                "request_id": request_id,
                "endpoint": "/health",
                "status_code": 500,
                "error": str(e)
            }
        )

        # Capture exception in Sentry
        if sentry_dsn:
            sentry_sdk.capture_exception(e)

        return HealthCheck(
            status="unhealthy",
            model_loaded=False,
            model_version=None
        )


@app.get("/model", response_model=ModelInfo)
async def get_model_info(request: Request):
    """
    Get information about the loaded model.

    Returns:
        ModelInfo: Detailed information about the model
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    try:
        if not prediction_service.is_ready():
            logger.error(
                "Model not loaded",
                extra={
                    "request_id": request_id,
                    "endpoint": "/model",
                    "status_code": 503
                }
            )
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Add Sentry context
        if sentry_dsn:
            with sentry_sdk.new_scope() as scope:
                scope.set_tag("endpoint", "/model")
                scope.set_tag("info_type", "model_metadata")

        model_info = prediction_service.get_model_info()

        logger.info(
            "Model info retrieved",
            extra={
                "request_id": request_id,
                "endpoint": "/model",
                "status_code": 200,
                "model_type": model_info["model_type"],
                "feature_count": len(model_info["features"])
            }
        )

        return ModelInfo(
            model_type=model_info["model_type"],
            model_version=model_info["model_version"],
            features=model_info["features"],
            training_date=None  # Could be added if available
        )
    except Exception as e:
        logger.error(
            f"Error getting model info: {str(e)}",
            extra={
                "request_id": request_id,
                "endpoint": "/model",
                "status_code": 500,
                "error": str(e)
            }
        )

        # Capture exception in Sentry
        if sentry_dsn:
            sentry_sdk.capture_exception(e)

        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@app.post("/predict", response_model=PredictionResult)
async def predict_churn(request: Request, customer_data: CustomerFeatures):
    """
    Predict customer churn based on input features.

    Args:
        request: FastAPI Request object for context
        customer_data: CustomerFeatures containing all required customer information

    Returns:
        PredictionResult: Prediction result with probability and metadata

    Raises:
        HTTPException: 400 if input validation fails or prediction error occurs
        HTTPException: 503 if model is not loaded
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    try:
        logger.info(
            "Received prediction request",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "customer_data": customer_data.model_dump()
            }
        )

        # Check if service is ready
        if not prediction_service.is_ready():
            logger.error(
                "Model not loaded",
                extra={
                    "request_id": request_id,
                    "endpoint": "/predict",
                    "status_code": 503
                }
            )
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Add Sentry context for this specific prediction
        if sentry_dsn:
            with sentry_sdk.new_scope() as scope:
                scope.set_tag("model_version", prediction_service.model_version)
                scope.set_tag("prediction_type", "churn")
                # Note: customer_data doesn't have customerID in our schema
                scope.set_extra("input_features", list(customer_data.model_dump().keys()))

        # Convert customer data to dict and make prediction
        input_data = customer_data.model_dump()
        churn_prediction, churn_probability = prediction_service.predict(input_data)

        logger.info(
            "Prediction completed",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "churn_prediction": churn_prediction,
                "churn_probability": churn_probability,
                "status_code": 200,
                "model_version": prediction_service.model_version
            }
        )

        return PredictionResult(
            churn_prediction=churn_prediction,
            churn_probability=churn_probability,
            model_version=prediction_service.model_version,
            request_id=request_id
        )

    except ValueError as ve:
        logger.error(
            f"Validation error: {str(ve)}",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "status_code": 400,
                "error_type": "ValidationError",
                "error_details": str(ve)
            }
        )
        raise HTTPException(status_code=400, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        logger.error(
            f"Prediction error: {str(e)}",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "status_code": 500,
                "error_type": "PredictionError",
                "error_details": str(e)
            }
        )

        # Capture exception in Sentry
        if sentry_dsn:
            sentry_sdk.capture_exception(e)

        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
