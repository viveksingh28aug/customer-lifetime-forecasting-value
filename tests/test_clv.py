"""Tests for Customer Lifetime Value (CLV) Models."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaner import clean_retail_data
from src.features.customer_features import compute_customer_metrics
from src.clv.baseline import compute_historical_baseline_clv, compute_transparent_forecast_clv
from src.clv.probabilistic import fit_probabilistic_clv
from src.clv.evaluation import summarize_clv_distribution

def test_historical_baseline_clv(sample_clean_transactions):
    """Test historical baseline CLV calculates positive annualized values."""
    sales_df, _, _ = clean_retail_data(sample_clean_transactions)
    cust_metrics = compute_customer_metrics(sales_df)
    clv_df = compute_historical_baseline_clv(cust_metrics, horizon_months=12)

    assert len(clv_df) == len(cust_metrics)
    assert "predicted_clv" in clv_df.columns
    assert (clv_df["predicted_clv"] >= 0).all()

def test_transparent_forecast_clv(sample_clean_transactions):
    """Test transparent forecast CLV integrates horizon, margin, and discount rate."""
    sales_df, _, _ = clean_retail_data(sample_clean_transactions)
    cust_metrics = compute_customer_metrics(sales_df)

    clv_df = compute_transparent_forecast_clv(
        cust_metrics,
        horizon_months=12,
        discount_rate=0.10,
        gross_margin=0.50,
        recency_decay_factor=True
    )

    assert len(clv_df) == len(cust_metrics)
    assert "predicted_clv" in clv_df.columns
    assert "predicted_transactions" in clv_df.columns
    assert "predicted_revenue" in clv_df.columns
    assert (clv_df["predicted_clv"] >= 0).all()
    assert (clv_df["gross_margin"] == 0.50).all()

def test_probabilistic_clv_fallback_on_small_data(sample_clean_transactions):
    """Test probabilistic model triggers transparent fallback when repeat customers < 10."""
    sales_df, _, _ = clean_retail_data(sample_clean_transactions)
    cust_metrics = compute_customer_metrics(sales_df)

    clv_df, diag = fit_probabilistic_clv(sales_df, cust_metrics, horizon_months=12)

    # Since sample_clean_transactions has only 2 repeat customers (< 10), it MUST fall back
    assert diag.is_probabilistic is False
    assert "Fallback" in diag.model_name
    assert diag.repeat_customers_count < 10
    assert len(clv_df) == len(cust_metrics)
    assert (clv_df["predicted_clv"] >= 0).all()

def test_clv_distribution_summary():
    """Test CLV distribution summary computes valid statistical aggregates."""
    test_df = pd.DataFrame({
        "CustomerID": ["1", "2", "3", "4", "5"],
        "predicted_clv": [100.0, 200.0, 300.0, 400.0, 1000.0]
    })
    summary = summarize_clv_distribution(test_df)
    assert summary["Total Predicted CLV"] == 2000.0
    assert summary["Mean CLV"] == 400.0
    assert summary["Median CLV"] == 300.0
    assert summary["Min CLV"] == 100.0
    assert summary["Max CLV"] == 1000.0
    assert summary["Top 10% Customer Value Share (%)"] > 0
