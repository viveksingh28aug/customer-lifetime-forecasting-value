"""Tests for Data Cleaning and Data Quality Auditing."""

import pandas as pd
import pytest

from src.preprocessing.cleaner import clean_retail_data
from src.preprocessing.data_quality import DataQualityReport

def test_cleaner_removes_duplicates(sample_messy_transactions):
    """Test duplicate transaction rows are identified and dropped."""
    sales_df, returns_df, dq_report = clean_retail_data(sample_messy_transactions, run_id="test_run")
    assert dq_report.duplicate_rows == 1

def test_cleaner_handles_missing_customer_ids(sample_messy_transactions):
    """Test records with missing CustomerID are excluded from sales and counted in DQ."""
    sales_df, returns_df, dq_report = clean_retail_data(sample_messy_transactions, run_id="test_run")
    assert dq_report.missing_customer_ids == 1
    # CustomerID in cleaned sales must never be null
    assert sales_df["CustomerID"].isna().sum() == 0

def test_cleaner_formats_float_customer_ids(sample_messy_transactions):
    """Test float-formatted CustomerID like 202.0 is converted to clean string '202'."""
    sales_df, returns_df, dq_report = clean_retail_data(sample_messy_transactions, run_id="test_run")
    assert "202" in sales_df["CustomerID"].values
    assert not any(".0" in str(cid) for cid in sales_df["CustomerID"])

def test_cleaner_isolates_cancellations_and_returns(sample_messy_transactions):
    """Test cancellation invoices (C-prefix / negative quantity) are separated from sales."""
    sales_df, returns_df, dq_report = clean_retail_data(sample_messy_transactions, run_id="test_run")
    assert dq_report.return_records == 1
    assert len(returns_df) == 1
    # Sales DataFrame must have exclusively positive quantities
    assert (sales_df["Quantity"] > 0).all()

def test_cleaner_drops_invalid_dates(sample_messy_transactions):
    """Test unparseable dates are dropped and logged in DQ report."""
    sales_df, returns_df, dq_report = clean_retail_data(sample_messy_transactions, run_id="test_run")
    assert dq_report.invalid_dates == 1
    # Dates in sales_df must all be valid datetimes
    assert pd.api.types.is_datetime64_any_dtype(sales_df["InvoiceDate"])

def test_cleaner_drops_non_positive_prices(sample_messy_transactions):
    """Test UnitPrice <= 0 records are excluded from sales."""
    sales_df, returns_df, dq_report = clean_retail_data(sample_messy_transactions, run_id="test_run")
    assert dq_report.invalid_unit_prices == 1
    assert (sales_df["UnitPrice"] > 0).all()

def test_cleaner_calculates_total_amount(sample_clean_transactions):
    """Test TotalAmount calculation equals Quantity * UnitPrice."""
    sales_df, returns_df, dq_report = clean_retail_data(sample_clean_transactions, run_id="test_run")
    assert "TotalAmount" in sales_df.columns
    expected = (sales_df["Quantity"] * sales_df["UnitPrice"]).round(2)
    pd.testing.assert_series_equal(sales_df["TotalAmount"], expected, check_names=False)

def test_cleaner_empty_dataframe():
    """Test cleaner handles empty input gracefully without raising unhandled errors."""
    sales_df, returns_df, dq_report = clean_retail_data(pd.DataFrame(), run_id="empty_run")
    assert sales_df.empty
    assert returns_df.empty
    assert dq_report.total_raw_rows == 0
    assert dq_report.data_quality_score == 0.0
