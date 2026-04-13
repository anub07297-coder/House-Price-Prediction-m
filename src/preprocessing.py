import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using median for numeric and mode for categorical features."""
    logging.info("Handling missing values")
    df = df.copy()
    for column in df.columns:
        if df[column].isna().sum() == 0:
            continue

        if df[column].dtype in ["float64", "int64"]:
            fill_value = df[column].median()
        else:
            fill_value = df[column].mode().iloc[0]

        logging.debug("Filling missing values for %s with %s", column, fill_value)
        df[column] = df[column].fillna(fill_value)

    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features using one-hot encoding."""
    logging.info("Encoding categorical features")
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_columns) == 0:
        return df

    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: StandardScaler | None = None,
):
    """Scale numeric features using StandardScaler."""
    logging.info("Scaling feature values")
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """Preprocess training and test feature sets."""
    X_train = handle_missing_values(X_train)
    X_test = handle_missing_values(X_test)

    X_train = encode_categorical(X_train)
    X_test = encode_categorical(X_test)

    # Keep only matching columns after encoding
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, scaler, X_train.columns.tolist()
