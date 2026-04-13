import logging
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model using RMSE and R² metrics."""
    logging.info("Evaluating model")
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    logging.info("RMSE: %s", rmse)
    logging.info("R²: %s", r2)
    return {"rmse": rmse, "r2": r2}
