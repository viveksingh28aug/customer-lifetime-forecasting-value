"""Monthly Revenue Forecasting Engine.

Implements monthly aggregation, strictly chronological train/test splitting,
Naive, Seasonal-Naive, and Holt-Winters Exponential Smoothing forecasting with evaluation metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.forecasting.evaluation import calculate_forecast_metrics
from config.logging_config import get_logger

logger = get_logger("src.forecasting.revenue_forecast")

@dataclass
class ForecastResults:
    """Encapsulates time-series forecasting artifacts and diagnostics."""
    monthly_history: pd.DataFrame
    test_evaluations: pd.DataFrame
    metrics_summary: pd.DataFrame
    future_forecasts: pd.DataFrame
    train_months: int
    test_months: int
    horizon_months: int
    notes: List[str]

def aggregate_monthly_revenue(df_sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregates transactional sales into a continuous monthly revenue time-series.

    Args:
        df_sales: Cleaned sales transactions with InvoiceDate and TotalAmount.

    Returns:
        DataFrame with continuous monthly PeriodIndex and TotalRevenue.
    """
    if df_sales.empty or "InvoiceDate" not in df_sales.columns or "TotalAmount" not in df_sales.columns:
        logger.warning("Empty or invalid sales data for monthly revenue aggregation.")
        return pd.DataFrame()

    df_work = df_sales.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work["InvoiceDate"]):
        df_work["InvoiceDate"] = pd.to_datetime(df_work["InvoiceDate"], errors="coerce")

    # Drop NaT
    df_work = df_work.dropna(subset=["InvoiceDate"])
    if df_work.empty:
        return pd.DataFrame()

    # Aggregate by month
    df_work["YearMonth"] = df_work["InvoiceDate"].dt.to_period("M")
    monthly = df_work.groupby("YearMonth").agg(
        TotalRevenue=("TotalAmount", "sum"),
        TotalInvoices=("InvoiceNo", "nunique"),
        UniqueCustomers=("CustomerID", "nunique")
    ).reset_index()

    monthly["TotalRevenue"] = monthly["TotalRevenue"].round(2)
    monthly = monthly.sort_values(by="YearMonth").reset_index(drop=True)

    # Reindex to ensure continuous monthly sequence with no gaps
    if len(monthly) > 1:
        full_idx = pd.period_range(start=monthly["YearMonth"].iloc[0], end=monthly["YearMonth"].iloc[-1], freq="M")
        monthly = monthly.set_index("YearMonth").reindex(full_idx, fill_value=0.0).reset_index()
        monthly = monthly.rename(columns={"index": "YearMonth"})

    # Convert YearMonth to timestamp for visualization and export
    monthly["PeriodDate"] = monthly["YearMonth"].dt.to_timestamp()

    logger.info(f"Aggregated monthly revenue across {len(monthly)} months.")
    return monthly

def fit_naive_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Projects the most recent observed actual forward."""
    last_val = history.iloc[-1] if len(history) > 0 else 0.0
    return np.full(horizon, last_val)

def fit_seasonal_naive_forecast(history: pd.Series, horizon: int) -> Tuple[np.ndarray, str]:
    """Projects historical seasonal observations forward when seasonal depth is sufficient."""
    n = len(history)
    if n >= 24:
        # Full 12-month lag seasonal naive
        forecast = []
        for step in range(horizon):
            lag_idx = n - 12 + (step % 12)
            forecast.append(history.iloc[lag_idx])
        return np.array(forecast), "12-Month Lag Seasonal Naive"
    elif n >= 12:
        # 12-month cycle repeat
        forecast = []
        for step in range(horizon):
            lag_idx = step % 12
            forecast.append(history.iloc[lag_idx])
        return np.array(forecast), "Single Cycle Seasonal Naive"
    else:
        # Fall back to naive
        return fit_naive_forecast(history, horizon), "Fallback to Naive (Insufficient history < 12M)"

def fit_holt_winters_forecast(
    history: pd.Series,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Fits Exponential Smoothing / Holt-Winters with defensible prediction intervals."""
    n = len(history)
    if n < 4:
        # Not enough data for exponential smoothing, use naive
        pred = fit_naive_forecast(history, horizon)
        return pred, pred, pred, "Fallback Naive (Insufficient history < 4M)"

    try:
        if n >= 24:
            # Full trend + seasonal model
            model = ExponentialSmoothing(
                history.values,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated"
            ).fit(optimized=True)
            desc = "Holt-Winters (Additive Trend & Additive Seasonality)"
        elif n >= 6:
            # Holt's linear trend model without seasonality
            model = ExponentialSmoothing(
                history.values,
                trend="add",
                seasonal=None,
                initialization_method="estimated"
            ).fit(optimized=True)
            desc = "Holt's Linear Trend (Additive Trend, No Seasonality)"
        else:
            # Simple exponential smoothing
            model = ExponentialSmoothing(
                history.values,
                trend=None,
                seasonal=None,
                initialization_method="estimated"
            ).fit(optimized=True)
            desc = "Simple Exponential Smoothing"

        forecast = model.forecast(horizon)
        # Prevent negative revenue forecast
        forecast = np.maximum(forecast, 0.0).round(2)

        # Defensible prediction interval using residual standard error
        residuals = history.values - model.fittedvalues
        residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0
        # 95% interval approximation (1.96 * se)
        margin = 1.96 * residual_std
        lower = np.maximum(forecast - margin, 0.0).round(2)
        upper = (forecast + margin).round(2)

        return forecast, lower, upper, desc

    except Exception as e:
        logger.warning(f"Holt-Winters optimization failed: {e}. Falling back to Naive.")
        pred = fit_naive_forecast(history, horizon)
        return pred, pred, pred, f"Fallback Naive (Exception: {str(e)[:50]})"

def run_revenue_forecast_pipeline(
    df_sales: pd.DataFrame,
    horizon_months: int = 6,
    test_months: Optional[int] = None,
    run_id: Optional[str] = None
) -> ForecastResults:
    """Executes chronological train/test validation and future multi-model revenue forecasting.

    Args:
        df_sales: Cleaned sales transactions.
        horizon_months: Future forecast horizon (3, 6, 12 months).
        test_months: Number of held-out months for chronological testing (defaults to 3 or 20%).
        run_id: Pipeline execution identifier.

    Returns:
        ForecastResults dataclass with history, evaluations, future projections, and metrics.
    """
    notes: List[str] = []
    monthly = aggregate_monthly_revenue(df_sales)

    if monthly.empty or len(monthly) < 3:
        logger.warning("Fewer than 3 months of revenue history. Forecasting is not reliable.")
        empty_metrics = pd.DataFrame([{
            "Model": "Insufficient Data",
            "MAE": None, "RMSE": None, "WAPE (%)": None, "MAPE (%)": None, "Observations": len(monthly)
        }])
        return ForecastResults(
            monthly_history=monthly,
            test_evaluations=pd.DataFrame(),
            metrics_summary=empty_metrics,
            future_forecasts=pd.DataFrame(),
            train_months=len(monthly),
            test_months=0,
            horizon_months=horizon_months,
            notes=["Dataset contains fewer than 3 months of history. Statistical forecasting requires more data."],
        )

    total_n = len(monthly)
    # Determine chronological test split
    if test_months is None:
        if total_n >= 12:
            test_size = 3
        elif total_n >= 6:
            test_size = 2
        else:
            test_size = 1
    else:
        test_size = min(test_months, total_n - 2)

    train_size = total_n - test_size
    train_df = monthly.iloc[:train_size].copy()
    test_df = monthly.iloc[train_size:].copy()

    logger.info(f"Chronological split: {train_size} train months, {test_size} test months.")

    # 1. Evaluate models on held-out test data
    y_train = train_df["TotalRevenue"]
    y_test = test_df["TotalRevenue"]

    # Model A: Naive
    pred_naive_test = fit_naive_forecast(y_train, test_size)
    metrics_naive = calculate_forecast_metrics(y_test, pd.Series(pred_naive_test, index=test_df.index))
    metrics_naive["Model"] = "Naive Baseline"

    # Model B: Seasonal-Naive
    pred_snaive_test, snaive_note = fit_seasonal_naive_forecast(y_train, test_size)
    metrics_snaive = calculate_forecast_metrics(y_test, pd.Series(pred_snaive_test, index=test_df.index))
    metrics_snaive["Model"] = f"Seasonal-Naive ({snaive_note})"

    # Model C: Holt-Winters
    pred_hw_test, _, _, hw_note = fit_holt_winters_forecast(y_train, test_size)
    metrics_hw = calculate_forecast_metrics(y_test, pd.Series(pred_hw_test, index=test_df.index))
    metrics_hw["Model"] = f"Holt-Winters ({hw_note})"

    metrics_df = pd.DataFrame([metrics_naive, metrics_snaive, metrics_hw])
    cols_order = ["Model", "MAE", "RMSE", "WAPE (%)", "MAPE (%)", "Observations"]
    metrics_df = metrics_df[[c for c in cols_order if c in metrics_df.columns]]

    # Prepare test evaluation comparison table
    test_eval_df = pd.DataFrame({
        "PeriodDate": test_df["PeriodDate"],
        "YearMonth": test_df["YearMonth"].astype(str),
        "ActualRevenue": test_df["TotalRevenue"].round(2),
        "Naive_Prediction": np.round(pred_naive_test, 2),
        "Seasonal_Naive_Prediction": np.round(pred_snaive_test, 2),
        "Holt_Winters_Prediction": np.round(pred_hw_test, 2),
    }).reset_index(drop=True)

    # 2. Generate Future Projections (Trained on Full Dataset)
    y_full = monthly["TotalRevenue"]
    last_period = monthly["YearMonth"].iloc[-1]
    future_periods = pd.period_range(start=last_period + 1, periods=horizon_months, freq="M")
    future_dates = future_periods.to_timestamp()

    # Model A Future
    fut_naive = fit_naive_forecast(y_full, horizon_months)
    # Model B Future
    fut_snaive, _ = fit_seasonal_naive_forecast(y_full, horizon_months)
    # Model C Future
    fut_hw, hw_lower, hw_upper, hw_desc = fit_holt_winters_forecast(y_full, horizon_months)

    future_records = []
    for i in range(horizon_months):
        p_date = future_dates[i]
        p_str = str(future_periods[i])

        # Naive row
        future_records.append({
            "PeriodDate": p_date,
            "YearMonth": p_str,
            "ModelName": "Naive Baseline",
            "ForecastHorizon": horizon_months,
            "PredictedRevenue": round(float(fut_naive[i]), 2),
            "LowerBound": round(float(fut_naive[i]), 2),
            "UpperBound": round(float(fut_naive[i]), 2),
            "IsTestActual": False,
            "ActualRevenue": None,
        })
        # Seasonal Naive row
        future_records.append({
            "PeriodDate": p_date,
            "YearMonth": p_str,
            "ModelName": "Seasonal-Naive",
            "ForecastHorizon": horizon_months,
            "PredictedRevenue": round(float(fut_snaive[i]), 2),
            "LowerBound": round(float(fut_snaive[i]), 2),
            "UpperBound": round(float(fut_snaive[i]), 2),
            "IsTestActual": False,
            "ActualRevenue": None,
        })
        # Holt-Winters row
        future_records.append({
            "PeriodDate": p_date,
            "YearMonth": p_str,
            "ModelName": "Holt-Winters",
            "ForecastHorizon": horizon_months,
            "PredictedRevenue": round(float(fut_hw[i]), 2),
            "LowerBound": round(float(hw_lower[i]), 2),
            "UpperBound": round(float(hw_upper[i]), 2),
            "IsTestActual": False,
            "ActualRevenue": None,
        })

    future_df = pd.DataFrame(future_records)
    if run_id:
        future_df["run_id"] = run_id
    future_df["created_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    notes.append(f"Validation executed using chronological split: {train_size} train months, {test_size} test months.")
    notes.append(f"Holt-Winters configuration: {hw_desc}.")
    if total_n < 24:
        notes.append(f"History has {total_n} months (< 24 months). True multi-year seasonality cannot be fully confirmed.")

    return ForecastResults(
        monthly_history=monthly,
        test_evaluations=test_eval_df,
        metrics_summary=metrics_df,
        future_forecasts=future_df,
        train_months=train_size,
        test_months=test_size,
        horizon_months=horizon_months,
        notes=notes,
    )
