"""Data Cleaning and Transformation Module.

Executes transparent data cleaning policies, separates cancellations/returns,
validates numerical constraints, formats customer identifiers, and computes Data Quality metrics.
"""

from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.preprocessing.data_quality import DataQualityReport
from config.logging_config import get_logger

logger = get_logger("src.preprocessing.cleaner")

def clean_retail_data(
    df: pd.DataFrame,
    run_id: Optional[str] = None,
    drop_duplicates: bool = True,
    drop_missing_customer: bool = True,
    drop_cancellations: bool = True,
    drop_non_positive_price: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, DataQualityReport]:
    """Cleans raw retail transaction records and generates a Data Quality audit report.

    Args:
        df: Raw transaction DataFrame adhering to standard columns.
        run_id: Unique pipeline run identifier.
        drop_duplicates: Whether to remove duplicate transaction rows.
        drop_missing_customer: Whether to exclude records with null/empty CustomerID.
        drop_cancellations: Whether to exclude returns and cancellations from primary sales.
        drop_non_positive_price: Whether to exclude UnitPrice <= 0.

    Returns:
        Tuple of (Cleaned Sales DataFrame, Returns/Cancellations DataFrame, DataQualityReport)
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame passed to cleaner.")
        empty_dq = DataQualityReport(
            run_id=run_id or "empty",
            total_raw_rows=0,
            total_valid_rows=0,
            duplicate_rows=0,
            missing_customer_ids=0,
            invalid_dates=0,
            return_records=0,
            invalid_unit_prices=0,
            excluded_rows=0,
            final_cleaned_rows=0,
            cleaning_policy="Empty Input",
            data_quality_score=0.0,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        return pd.DataFrame(), pd.DataFrame(), empty_dq

    raw_len = len(df)
    logger.info(f"Starting data cleaning for {raw_len} raw records (run_id={run_id})...")

    df_work = df.copy()

    # 1. Duplicates Detection & Removal
    dup_mask = df_work.duplicated()
    num_duplicates = int(dup_mask.sum())
    if drop_duplicates:
        df_work = df_work[~dup_mask].copy()

    # 2. Date Parsing
    df_work["InvoiceDate"] = pd.to_datetime(df_work["InvoiceDate"], errors="coerce")
    invalid_dates_mask = df_work["InvoiceDate"].isna()
    num_invalid_dates = int(invalid_dates_mask.sum())
    df_work = df_work[~invalid_dates_mask].copy()

    # 3. Customer ID Standardization & Missing Checks
    # Format CustomerID: remove trailing .0 if stored as float, treat nan/empty
    def format_customer_id(val):
        if pd.isna(val) or val is None or str(val).strip() in ["", "nan", "None"]:
            return np.nan
        val_str = str(val).strip()
        if val_str.endswith(".0"):
            val_str = val_str[:-2]
        return val_str

    df_work["CustomerID"] = df_work["CustomerID"].apply(format_customer_id)
    missing_cust_mask = df_work["CustomerID"].isna()
    num_missing_cust = int(missing_cust_mask.sum())

    # 4. Cancellation & Return Detection
    invoice_str = df_work["InvoiceNo"].astype(str).str.strip()
    qty_series = pd.to_numeric(df_work["Quantity"], errors="coerce").fillna(0)
    df_work["Quantity"] = qty_series

    # A return is identified by invoice starting with 'C' or 'c', or quantity < 0
    return_mask = invoice_str.str.startswith("C", na=False) | invoice_str.str.startswith("c", na=False) | (df_work["Quantity"] < 0)
    num_returns = int(return_mask.sum())
    df_work["is_return"] = return_mask

    # 5. Unit Price Validation
    price_series = pd.to_numeric(df_work["UnitPrice"], errors="coerce").fillna(0.0)
    df_work["UnitPrice"] = price_series
    invalid_price_mask = df_work["UnitPrice"] <= 0
    num_invalid_prices = int(invalid_price_mask.sum())

    # 6. Separate Returns/Cancellations DataFrame
    returns_df = df_work[df_work["is_return"]].copy()

    # 7. Apply Cleaning Policies to isolate primary Sales Transactions
    sales_mask = pd.Series(True, index=df_work.index)

    if drop_missing_customer:
        sales_mask &= ~missing_cust_mask
    if drop_cancellations:
        sales_mask &= ~df_work["is_return"]
    if drop_non_positive_price:
        sales_mask &= ~invalid_price_mask

    # Ensure positive quantity for sales
    sales_mask &= (df_work["Quantity"] > 0)

    sales_df = df_work[sales_mask].copy()

    # Compute TotalAmount for sales
    sales_df["TotalAmount"] = (sales_df["Quantity"] * sales_df["UnitPrice"]).round(2)

    # Clean text columns
    sales_df["Description"] = sales_df["Description"].fillna("Unknown Item").astype(str).str.strip()
    sales_df["Country"] = sales_df["Country"].fillna("Unspecified").astype(str).str.strip()
    sales_df["StockCode"] = sales_df["StockCode"].astype(str).str.strip()
    sales_df["InvoiceNo"] = sales_df["InvoiceNo"].astype(str).str.strip()

    final_cleaned_len = len(sales_df)
    excluded_rows = raw_len - final_cleaned_len
    health_score = round((final_cleaned_len / raw_len * 100), 2) if raw_len > 0 else 0.0

    policy_summary = (
        f"DropDuplicates={drop_duplicates}, "
        f"DropMissingCustomer={drop_missing_customer}, "
        f"DropCancellations={drop_cancellations}, "
        f"DropNonPositivePrice={drop_non_positive_price}"
    )

    dq_report = DataQualityReport(
        run_id=run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        total_raw_rows=raw_len,
        total_valid_rows=final_cleaned_len + len(returns_df),
        duplicate_rows=num_duplicates,
        missing_customer_ids=num_missing_cust,
        invalid_dates=num_invalid_dates,
        return_records=num_returns,
        invalid_unit_prices=num_invalid_prices,
        excluded_rows=excluded_rows,
        final_cleaned_rows=final_cleaned_len,
        cleaning_policy=policy_summary,
        data_quality_score=health_score,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    logger.info(
        f"Cleaning complete: {final_cleaned_len} valid sales rows retained, "
        f"{num_returns} returns isolated, {excluded_rows} excluded. Quality score: {health_score}%"
    )

    return sales_df, returns_df, dq_report
