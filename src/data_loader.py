import logging
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the housing dataset from a CSV file."""
    logging.info("Loading data from %s", csv_path)
    try:
        df = pd.read_csv(csv_path)
        logging.debug("Loaded data with shape %s", df.shape)
        return df
    except FileNotFoundError as exc:
        logging.error("File not found: %s", csv_path)
        raise
    except Exception as exc:
        logging.error("Error loading data: %s", exc)
        raise


def split_data(
    df: pd.DataFrame,
    target_col: str = "median_house_value",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split the data into training and test sets."""
    logging.info("Splitting data into train and test sets")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
