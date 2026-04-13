import logging
from pathlib import Path
import joblib
import pandas as pd


def load_saved_model(model_path: str = "models/saved_model.pkl"):
    """Load a saved model and associated scaler and feature columns from disk."""
    logging.info("Loading saved model from %s", model_path)
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found at {model_path}")

    data = joblib.load(path)
    return data["model"], data["scaler"], data.get("feature_columns")


def prepare_new_data(data: pd.DataFrame, feature_columns):
    """Prepare new data for prediction by matching the training feature columns."""
    logging.info("Preparing new data for prediction")
    data = data.copy()
    data = pd.get_dummies(data)
    data = data.reindex(columns=feature_columns, fill_value=0)
    return data


def predict_new_data(model, scaler, data: pd.DataFrame, feature_columns):
    """Predict house prices for new data using the trained model."""
    logging.info("Predicting new data")
    if scaler is None:
        raise ValueError("Scaler is required to transform feature values before prediction")

    data_prepared = prepare_new_data(data, feature_columns)
    data_scaled = scaler.transform(data_prepared)
    return model.predict(data_scaled)
