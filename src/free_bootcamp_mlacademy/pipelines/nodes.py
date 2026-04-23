import pandas as pd
from typing import Dict, Any, Tuple, Union
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path

def rename_columns(df: pd.DataFrame, renaming_dict: Dict[str, str]) -> pd.DataFrame:
    """Rename columns based on column mapping."""
    return df.rename(columns=renaming_dict)


def get_features(df: pd.DataFrame, lag_params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Create lag features for time-series data.

    Generates lagged versions of specified columns, useful for capturing
    temporal patterns in time-series forecasting. Missing values at the
    beginning of each lagged column are backfilled.

    Args:
        df: Input DataFrame containing the original features.
        lag_params: Dictionary mapping feature names to lists of lag values.
            Example: {"temperature": [1, 2, 3]} creates columns
            "temperature_lag_1", "temperature_lag_2", "temperature_lag_3".

    Returns:
        DataFrame with original columns plus new lag feature columns.
    """
    for feature, lags in lag_params.items():
        for lag in lags:
            df[f"{feature}_lag_{lag}"] = df[feature].shift(lag).bfill()
    timestamps = pd.to_datetime(df['datetime'])
    df.drop(columns=['datetime'], inplace=True)
    return df, timestamps


def make_target(df: pd.DataFrame, target_params: Dict[str, Any]) -> pd.DataFrame:
    """Create target column by shifting."""
    df[target_params["new_target_name"]] = (
        df[target_params["target_column"]].shift(-target_params["shift_period"]).ffill()
    )
    return df


def split_data(
    df: pd.DataFrame, 
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test sets."""
    # Get target column name
    target_name = params["target_params"]["new_target_name"]
    # Get features columns names
    features = [col for col in df.columns if col != target_name]
    # Split data into train/test sets
    x, y = df[features], df[target_name]
    train_size = int(params["train_fraction"] * len(df))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return x_train, x_test, y_train, y_test


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
) -> Any:
    """Train a regression model with specified parameters.

    Supports multiple model types that can be selected via the params dict.
    Model type matching is case-insensitive and ignores leading/trailing whitespace.

    Args:
        x_train: Training features DataFrame.
        y_train: Training target Series.
        params: Dictionary containing:
            - model_type: Type of model to train. Supported values:
                - "catboost" or "cb": CatBoost gradient boosting
                - "random_forest" or "rf": Scikit-learn Random Forest
                - "linear_regression" or "linreg": Scikit-learn Linear Regression
            - model_params: Nested dict with params for each model type.

    Returns:
        Trained model instance
    """
    # Normalize model type: lowercase and remove whitespace
    model_type = params["model_type"].lower().strip()

    # Get model-specific parameters
    model_params = params["model_params"][model_type]

    # Select and instantiate the appropriate model
    if model_type in ["catboost", "cb"]:
        model = CatBoostRegressor(**model_params)
    elif model_type in ["random_forest", "rf"]:
        model = RandomForestRegressor(**model_params)
    elif model_type in ["linear_regression", "linreg"]:
        model = LinearRegression(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Fit model on training data
    model.fit(x_train, y_train)
    return model

def predict(
    model: Any,
    x: pd.DataFrame,
) -> pd.DataFrame:
    """Predict using a trained model."""
    y_pred = pd.DataFrame(model.predict(x), columns=["prediction"])
    print(f"Predictions {y_pred}")
    return y_pred


def compute_metrics(
    y_true: Union[np.ndarray, list], 
    y_pred: Union[np.ndarray, list]
) -> Dict[str, float]:
    """
    Compute evaluation metrics between true and predicted values.

    Metrics returned:
    - MAPE: Mean Absolute Percentage Error (in %)
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error

    Parameters:
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns:
    -------
    dict
        Dictionary with keys 'MAPE', 'MAE', and 'RMSE' and their float values.
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mape = np.mean(np.abs((y_true - y_pred) / y_true + 1e-8)) * 100
    
    metrics = {
        'MAE': float(round(mae, 2)),
        'RMSE': float(round(rmse, 2)),
        'MAPE': float(round(mape, 2)),
    }
    print(f"Metrics {metrics}")
    return metrics

def save_model(
    model: Any,
    model_type: str,
    model_storage: Dict[str, Any],
) -> None:
    """Persist the trained model to disk.

    Uses model-specific serialization when available:
    - CatBoost: native .cbm format
    - Other models: joblib .pkl format

    Args:
        model: Trained model instance.
        model_type: Type of model (for determining save format).
        model_storage: Dictionary containing:
            - path: Directory path where to save the model.
            - name: Model file name (without extension).
    """
    model_dir = Path(model_storage["path"])
    model_name = model_storage["name"]
    model_type = model_type.lower().strip()

    if model_type in ["catboost", "cb"]:
        # CatBoost has native serialization
        model.save_model(str(model_dir / f"{model_name}.cbm"))
    else:
        # Use joblib for sklearn models
        joblib.dump(model, model_dir / f"{model_name}.pkl")
    return None


def load_model(
    model_type: str,
    model_storage: Dict[str, Any],
) -> Any:
    """Load a model from disk.

    Uses model-specific deserialization:
    - CatBoost: native .cbm format
    - Other models: joblib .pkl format

    Args:
        model_type: Type of model (for determining load format).
        model_storage: Dictionary containing:
            - path: Directory path where the model is stored.
            - name: Model file name (without extension).

    Returns:
        Loaded model instance.
    """
    model_dir = Path(model_storage["path"])
    model_name = model_storage["name"]
    model_type = model_type.lower().strip()

    if model_type in ["catboost", "cb"]:
        # CatBoost has native deserialization
        model = CatBoostRegressor()
        model.load_model(str(model_dir / f"{model_name}.cbm"))
    else:
        # Use joblib for sklearn models
        model = joblib.load(model_dir / f"{model_name}.pkl")

    return model


def load_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Load data and extract last timestamp."""
    last_timestamp = pd.to_datetime(df["datetime"]).iloc[-1]
    return df, last_timestamp


def join_timestamps(
    predictions: pd.DataFrame,
    timestamps: pd.Timestamp,
) -> pd.DataFrame:
    """Join timestamp to predictions."""
    predictions["datetime"] = timestamps
    return predictions