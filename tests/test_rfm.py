"""Tests for Customer Aggregation and RFM Segmentation."""

from datetime import datetime
import pandas as pd
import pytest

from src.preprocessing.cleaner import clean_retail_data
from src.features.customer_features import compute_customer_metrics
from src.segmentation.rfm import perform_rfm_segmentation, classify_rfm_segment, compute_quantile_score

def test_customer_metrics_aggregation(sample_clean_transactions):
    """Test customer aggregation computes correct invoices, total spend, and lifespans."""
    sales_df, _, _ = clean_retail_data(sample_clean_transactions)
    cust_metrics = compute_customer_metrics(sales_df, reference_date=datetime(2024, 5, 1))

    assert len(cust_metrics) == 4
    assert "CustomerID" in cust_metrics.columns
    assert "num_invoices" in cust_metrics.columns
    assert "total_revenue" in cust_metrics.columns

    # Customer 101 placed 3 orders: 2*10=20, 1*30=30, 4*25=100 -> total 150.0
    c101 = cust_metrics[cust_metrics["CustomerID"] == "101"].iloc[0]
    assert c101["num_invoices"] == 3
    assert c101["total_revenue"] == 150.0
    assert c101["avg_order_value"] == 50.0
    assert c101["customer_lifespan_days"] > 0
    assert c101["avg_interpurchase_time"] > 0

def test_single_purchase_customers(sample_clean_transactions):
    """Test single-purchase customer has lifespan=0 and interpurchase_time=0 without division by zero."""
    sales_df, _, _ = clean_retail_data(sample_clean_transactions)
    cust_metrics = compute_customer_metrics(sales_df, reference_date=datetime(2024, 5, 1))

    # Customer 103 has 1 purchase
    c103 = cust_metrics[cust_metrics["CustomerID"] == "103"].iloc[0]
    assert c103["num_invoices"] == 1
    assert c103["customer_lifespan_days"] == 0.0
    assert c103["avg_interpurchase_time"] == 0.0

def test_rfm_quantile_scoring_handles_ties():
    """Test quantile scoring functions without error when series contains tied values."""
    tied_series = pd.Series([1, 1, 1, 1, 2, 5, 10, 20])
    scores = compute_quantile_score(tied_series, num_bins=5, ascending=True)
    assert len(scores) == 8
    assert scores.min() >= 1
    assert scores.max() <= 5

def test_rfm_segment_classification_rules():
    """Test deterministic segment classification logic for known RFM score combinations."""
    assert classify_rfm_segment(5, 5, 5) == "Champions"
    assert classify_rfm_segment(4, 4, 4) == "Champions"
    assert classify_rfm_segment(4, 1, 2) == "Recent Customers"
    assert classify_rfm_segment(4, 2, 3) == "Potential Loyalists"
    assert classify_rfm_segment(3, 3, 3) == "Loyal Customers"
    assert classify_rfm_segment(1, 4, 5) == "At-Risk Customers"
    assert classify_rfm_segment(1, 1, 1) == "Hibernating Customers"

def test_rfm_segmentation_summary_table(sample_clean_transactions):
    """Test perform_rfm_segmentation creates valid scores, segment names, and summary."""
    sales_df, _, _ = clean_retail_data(sample_clean_transactions)
    cust_metrics = compute_customer_metrics(sales_df)
    df_rfm, segment_summary = perform_rfm_segmentation(cust_metrics)

    assert len(df_rfm) == len(cust_metrics)
    assert "segment" in df_rfm.columns
    assert "rfm_score" in df_rfm.columns
    assert len(segment_summary) == 7  # All 7 documented segments represented in summary
    assert "Total Revenue" in segment_summary.columns
    assert "Customer Share (%)" in segment_summary.columns
