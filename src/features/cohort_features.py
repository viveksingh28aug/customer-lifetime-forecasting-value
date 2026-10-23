"""Cohort Features and Retention Analysis Module.

Builds customer acquisition cohort matrices and calculates month-by-month cohort retention rates.
"""

from typing import Tuple
import pandas as pd

from config.logging_config import get_logger

logger = get_logger("src.features.cohort_features")

def create_cohort_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates monthly cohort customer counts and percentage retention matrices.

    Args:
        df: Cleaned sales transactions with CustomerID and InvoiceDate.

    Returns:
        Tuple of (Cohort Counts Matrix DataFrame, Cohort Retention Percentage Matrix DataFrame)
    """
    if df.empty or "CustomerID" not in df.columns or "InvoiceDate" not in df.columns:
        logger.warning("Insufficient data for cohort matrix calculation.")
        return pd.DataFrame(), pd.DataFrame()

    df_work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work["InvoiceDate"]):
        df_work["InvoiceDate"] = pd.to_datetime(df_work["InvoiceDate"], errors="coerce")

    # Transaction period (Year-Month)
    df_work["InvoiceMonth"] = df_work["InvoiceDate"].dt.to_period("M")

    # Customer cohort month (Earliest purchase period)
    customer_cohort = df_work.groupby("CustomerID")["InvoiceMonth"].min().reset_index()
    customer_cohort = customer_cohort.rename(columns={"InvoiceMonth": "CohortMonth"})

    # Merge cohort month back
    df_merged = df_work.merge(customer_cohort, on="CustomerID", how="left")

    # Compute cohort index (number of months since acquisition)
    invoice_year = df_merged["InvoiceMonth"].dt.year
    invoice_month = df_merged["InvoiceMonth"].dt.month
    cohort_year = df_merged["CohortMonth"].dt.year
    cohort_month = df_merged["CohortMonth"].dt.month

    df_merged["CohortIndex"] = (invoice_year - cohort_year) * 12 + (invoice_month - cohort_month)

    # Group by CohortMonth and CohortIndex
    cohort_grouped = df_merged.groupby(["CohortMonth", "CohortIndex"])["CustomerID"].nunique().reset_index()

    # Pivot to create cohort matrix
    cohort_counts = cohort_grouped.pivot(
        index="CohortMonth",
        columns="CohortIndex",
        values="CustomerID"
    )

    # Sort cohort months chronologically
    cohort_counts = cohort_counts.sort_index()

    # Calculate retention percentage relative to month 0
    cohort_sizes = cohort_counts.iloc[:, 0]
    cohort_retention = cohort_counts.divide(cohort_sizes, axis=0) * 100.0
    cohort_retention = cohort_retention.round(1)

    # Format index as string for display and serialization
    cohort_counts.index = cohort_counts.index.astype(str)
    cohort_retention.index = cohort_retention.index.astype(str)

    logger.info(f"Generated cohort matrix: {len(cohort_counts)} cohorts across {len(cohort_counts.columns)} periods.")
    return cohort_counts, cohort_retention
