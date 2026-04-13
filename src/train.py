import logging
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def train_model(X_train, y_train, model_type: str = "random_forest"):
    """Train a regression model on the preprocessed training data."""
    logging.info("Training model: %s", model_type)
    if model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    model.fit(X_train, y_train)
    logging.info("Model training complete")
    return model


def save_model(model, scaler, feature_columns, output_path: str = "models/saved_model.pkl"):
    """Save the trained model, scaler, and feature layout to disk."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Saving model to %s", output_file)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
        },
        output_file,
    )
