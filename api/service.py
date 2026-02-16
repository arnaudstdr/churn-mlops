import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class ChurnPredictionService:
    """
    Service for loading model and preprocessor, and making churn predictions.
    """

    def __init__(self, model_path: Optional[str] = None, preprocessor_path: Optional[str] = None):
        """
        Initialize the prediction service.

        Args:
            model_path: Path to the trained model file
            preprocessor_path: Path to the preprocessor file
        """
        # Set default paths if None is provided
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "models" / "logistic_regression_model.joblib")
        if preprocessor_path is None:
            preprocessor_path = str(Path(__file__).parent.parent / "models" / "preprocessor.joblib")

        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.model_version = "1.0.0"  # This should be dynamically loaded or configured
        self._load_artefacts()

    def _get_default_model_path(self) -> str:
        """Get the default path to the model file."""
        return str(Path(__file__).parent.parent / "models" / "logistic_regression_model.joblib")

    def _get_default_preprocessor_path(self) -> str:
        """Get the default path to the preprocessor file."""
        return str(Path(__file__).parent.parent / "models" / "preprocessor.joblib")

    def _load_artefacts(self) -> None:
        """Load the model and preprocessor from files."""
        try:
            # Load model
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("Model loaded successfully")
            else:
                logger.error(f"Model file not found at {self.model_path}")
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

            # Load preprocessor
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                logger.info("Preprocessor loaded successfully")
            else:
                logger.error(f"Preprocessor file not found at {self.preprocessor_path}")
                raise FileNotFoundError(f"Preprocessor file not found at {self.preprocessor_path}")

        except Exception as e:
            logger.error(f"Error loading artefacts: {str(e)}")
            raise

    def preprocess_input(self, input_data: dict):
        """
        Convert input dictionary to DataFrame and apply preprocessing.

        Args:
            input_data: Dictionary containing customer features

        Returns:
            Preprocessed data ready for prediction (numpy array or sparse matrix)
        """
        try:
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Apply preprocessing
            if self.preprocessor:
                processed_data = self.preprocessor.transform(input_df)
                return processed_data
            else:
                raise ValueError("Preprocessor not loaded")

        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise

    def predict(self, input_data: dict, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Make a churn prediction for a given customer.

        Args:
            input_data: Dictionary containing customer features
            threshold: Probability threshold for churn prediction (default: 0.5)

        Returns:
            Tuple containing (churn_prediction, churn_probability)
        """
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)

            # Make prediction
            if self.model:
                # Get probability of churn (class 1)
                probabilities = self.model.predict_proba(processed_data)
                churn_probability = float(probabilities[0, 1])  # Probability of class 1 (churn)

                # Apply threshold
                churn_prediction = churn_probability >= threshold

                return churn_prediction, churn_probability
            else:
                raise ValueError("Model not loaded")

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        if not self.model or not self.preprocessor:
            return {
                "model_loaded": False,
                "error": "Model or preprocessor not loaded"
            }

        try:
            # Safely get feature names if available
            features = []
            if hasattr(self.preprocessor, 'named_transformers_') and \
               'cat' in self.preprocessor.named_transformers_ and \
               hasattr(self.preprocessor.named_transformers_['cat'], 'named_steps') and \
               'onehot' in self.preprocessor.named_transformers_['cat'].named_steps and \
               hasattr(self.preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
                features = list(self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out())

            return {
                "model_type": "LogisticRegression",
                "model_version": self.model_version,
                "features": features,
                "model_loaded": True
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {
                "model_loaded": False,
                "error": f"Error getting model info: {str(e)}"
            }

    def is_ready(self) -> bool:
        """
        Check if the service is ready to make predictions.

        Returns:
            True if both model and preprocessor are loaded, False otherwise
        """
        return self.model is not None and self.preprocessor is not None
