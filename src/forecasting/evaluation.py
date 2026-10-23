"""Time-Series Forecast Evaluation Metrics Module.

Calculates chronological evaluation metrics: MAE, RMSE, WAPE, and MAPE with safe zero handling.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger("src.forecasting.evaluation")

def calculate_forecast_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Optional[float]]:
    """Calculates defensible forecast evaluation metrics on held-out actuals.

    Args:
        y_true: Ground truth observed series.
        y_pred: Model predicted series aligned with y_true.

    Returns:
        Dictionary containing MAE, RMSE, WAPE, and MAPE.
    """
    if y_true is None or y_pred is None or len(y_true) == 0:
        return {"MAE": None, "RMSE": None, "WAPE": None, "MAPE": None, "Observations": 0}

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    # Filter out NaNs
    valid_mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    if not np.any(valid_mask):
        return {"MAE": None, "RMSE": None, "WAPE": None, "MAPE": None, "Observations": 0}

    y_t = y_t[valid_mask]
    y_p = y_p[valid_mask]
    n_obs = len(y_t)

    # 1. Mean Absolute Error
    mae = float(np.mean(np.abs(y_t - y_p)))

    # 2. Root Mean Squared Error
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    # 3. Weighted Absolute Percentage Error (WAPE = sum(|y - y_hat|) / sum(y) * 100)
    # Robust against individual periods with near-zero revenue
    sum_actual = float(np.sum(np.abs(y_t)))
    wape = float((np.sum(np.abs(y_t - y_p)) / sum_actual) * 100.0) if sum_actual > 0 else None

    # 4. Mean Absolute Percentage Error (MAPE)
    # Exclude zero values to prevent division by zero
    non_zero = y_t != 0
    if np.sum(non_zero) > 0:
        mape = float(np.mean(np.abs((y_t[non_zero] - y_p[non_zero]) / y_t[non_zero])) * 100.0)
    else:
        mape = None

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE (%)": round(wape, 2) if wape is not None else None,
        "MAPE (%)": round(mape, 2) if mape is not None else None,
        "Observations": n_obs,
    }
