"""Customer-Level Feature Aggregation Module.

Transforms transaction-level history into rich customer-level behavioral metrics:
frequency, monetary value, order history, lifespan, and interpurchase timing.
"""

from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger("src.features.customer_features")

def compute_customer_metrics(
    df: pd.DataFrame,
    reference_date: Optional[datetime] = None,
    run_id: Optional[str] = None
) -> pd.DataFrame:
    """Aggregates transaction data into comprehensive customer-level metrics.

    Args:
        df: Cleaned transaction records containing CustomerID, InvoiceNo, InvoiceDate, Quantity, TotalAmount.
        reference_date: Cutoff date for recency calculation (defaults to dataset max date + 1 day).
        run_id: Optional pipeline execution identifier.

    Returns:
        DataFrame indexed by CustomerID with granular customer behavioral metrics.
    """
    if df.empty or "CustomerID" not in df.columns:
        logger.warning("Empty DataFrame or missing CustomerID in customer metric aggregation.")
        return pd.DataFrame()

    df_work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work["InvoiceDate"]):
        df_work["InvoiceDate"] = pd.to_datetime(df_work["InvoiceDate"], errors="coerce")

    # Set reference date to 1 day after the latest transaction in data if not specified
    if reference_date is None:
        reference_date = df_work["InvoiceDate"].max() + timedelta(days=1)

    logger.info(f"Computing customer metrics for {df_work['CustomerID'].nunique()} unique customers (reference_date={reference_date}).")

    # Group by customer
    # Note: Distinct invoices must be counted per customer, NOT rows!
    customer_agg = df_work.groupby("CustomerID").agg(
        first_purchase_date=("InvoiceDate", "min"),
        last_purchase_date=("InvoiceDate", "max"),
        num_invoices=("InvoiceNo", "nunique"),
        total_quantity=("Quantity", "sum"),
        total_revenue=("TotalAmount", "sum"),
    ).reset_index()

    # Recency (Days since last purchase relative to reference_date)
    customer_agg["recency_days"] = (
        (reference_date - customer_agg["last_purchase_date"]).dt.total_seconds() / 86400.0
    ).round(2)

    # Lifespan (Days between first and last purchase)
    customer_agg["customer_lifespan_days"] = (
        (customer_agg["last_purchase_date"] - customer_agg["first_purchase_date"]).dt.total_seconds() / 86400.0
    ).round(2)

    # Average Order Value (AOV)
    customer_agg["avg_order_value"] = (
        customer_agg["total_revenue"] / customer_agg["num_invoices"].replace(0, 1)
    ).round(2)
    customer_agg["avg_revenue_per_purchase"] = customer_agg["avg_order_value"]

    # Average Interpurchase Time in days
    # For single-purchase customers, interpurchase time is 0.0 (or NaN handled cleanly)
    repeat_mask = customer_agg["num_invoices"] > 1
    customer_agg["avg_interpurchase_time"] = 0.0
    customer_agg.loc[repeat_mask, "avg_interpurchase_time"] = (
        customer_agg.loc[repeat_mask, "customer_lifespan_days"] / (customer_agg.loc[repeat_mask, "num_invoices"] - 1)
    ).round(2)

    # Monthly purchase frequency: orders per month of active lifespan (minimum 1 month base)
    active_months = np.maximum(customer_agg["customer_lifespan_days"] / 30.44, 1.0)
    customer_agg["purchase_frequency"] = (customer_agg["num_invoices"] / active_months).round(3)

    if run_id:
        customer_agg["run_id"] = run_id
    customer_agg["calculated_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sort by total revenue descending
    customer_agg = customer_agg.sort_values(by="total_revenue", ascending=False).reset_index(drop=True)

    logger.info(f"Customer metrics calculated successfully for {len(customer_agg)} customers.")
    return customer_agg
