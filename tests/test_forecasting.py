"""Tests for Monthly Revenue Time-Series Forecasting and Evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.revenue_forecast import (
    aggregate_monthly_revenue,
    fit_naive_forecast,
    fit_seasonal_naive_forecast,
    fit_holt_winters_forecast,
    run_revenue_forecast_pipeline,
)
from src.forecasting.evaluation import calculate_forecast_metrics

def test_aggregate_monthly_revenue(sample_clean_transactions):
    """Test monthly aggregation creates continuous sequence of monthly revenue totals."""
    sample_clean_transactions["TotalAmount"] = (
        sample_clean_transactions["Quantity"] * sample_clean_transactions["UnitPrice"]
    )
    monthly = aggregate_monthly_revenue(sample_clean_transactions)

    assert not monthly.empty
    assert "YearMonth" in monthly.columns
    assert "TotalRevenue" in monthly.columns
    assert (monthly["TotalRevenue"] > 0).any()
    # Check chronological ordering
    assert list(monthly["YearMonth"]) == sorted(list(monthly["YearMonth"]))

def test_forecast_evaluation_metrics_calculation():
    """Test MAE, RMSE, WAPE, and MAPE metrics calculations against known synthetic series."""
    y_true = pd.Series([100.0, 200.0, 300.0])
    y_pred = pd.Series([110.0, 190.0, 310.0])

    metrics = calculate_forecast_metrics(y_true, y_pred)
    assert metrics["MAE"] == 10.0
    assert metrics["RMSE"] == 10.0
    # WAPE: (|10| + |-10| + |10|) / (100 + 200 + 300) = 30 / 600 = 5%
    assert metrics["WAPE (%)"] == 5.0
    assert metrics["Observations"] == 3

def test_forecast_evaluation_handles_zero_actuals():
    """Test evaluation handles zero actual without crashing or zero-division exception."""
    y_true = pd.Series([0.0, 100.0])
    y_pred = pd.Series([10.0, 90.0])

    metrics = calculate_forecast_metrics(y_true, y_pred)
    assert metrics["MAE"] == 10.0
    assert metrics["RMSE"] == 10.0
    # MAPE should only evaluate non-zero observations
    assert metrics["MAPE (%)"] == 10.0

def test_chronological_train_test_split_integrity():
    """Test forecasting uses strictly chronological train/test split without lookahead leakage."""
    dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    amounts = [100.0, 110.0, 105.0, 120.0, 115.0, 130.0, 125.0, 140.0, 135.0, 150.0, 145.0, 160.0]
    test_sales = pd.DataFrame({
        "InvoiceNo": [f"INV_{i}" for i in range(12)],
        "StockCode": ["A"] * 12,
        "Description": ["Item"] * 12,
        "Quantity": [10] * 12,
        "InvoiceDate": dates,
        "UnitPrice": [10.0] * 12,
        "CustomerID": [f"C_{i}" for i in range(12)],
        "Country": ["UK"] * 12,
        "TotalAmount": amounts,
    })

    results = run_revenue_forecast_pipeline(test_sales, horizon_months=3, test_months=3)
    assert results.train_months == 9
    assert results.test_months == 3
    assert len(results.test_evaluations) == 3
    assert not results.future_forecasts.empty

def test_insufficient_history_forecasting():
    """Test pipeline gracefully handles history under 3 months."""
    short_sales = pd.DataFrame({
        "InvoiceNo": ["INV_1", "INV_2"],
        "StockCode": ["A", "B"],
        "Description": ["A", "B"],
        "Quantity": [1, 2],
        "InvoiceDate": pd.to_datetime(["2024-01-01", "2024-01-15"]),
        "UnitPrice": [10.0, 20.0],
        "CustomerID": ["1", "2"],
        "Country": ["UK", "UK"],
        "TotalAmount": [10.0, 40.0],
    })

    results = run_revenue_forecast_pipeline(short_sales, horizon_months=3)
    assert results.test_months == 0
    assert any("fewer than 3 months" in n for n in results.notes)
