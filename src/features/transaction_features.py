"""Transaction-Level Feature Engineering Module.

Extracts temporal, basket-level, and line-item behavioral attributes from cleaned transactions.
"""

import pandas as pd

from config.logging_config import get_logger

logger = get_logger("src.features.transaction_features")

def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches cleaned transaction records with calendar, time, and revenue attributes.

    Args:
        df: Cleaned transaction DataFrame containing InvoiceDate, Quantity, UnitPrice.

    Returns:
        DataFrame enriched with temporal features.
    """
    if df.empty:
        return df

    df_feat = df.copy()

    # Ensure InvoiceDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_feat["InvoiceDate"]):
        df_feat["InvoiceDate"] = pd.to_datetime(df_feat["InvoiceDate"], errors="coerce")

    # Temporal features
    df_feat["Year"] = df_feat["InvoiceDate"].dt.year
    df_feat["Month"] = df_feat["InvoiceDate"].dt.month
    df_feat["YearMonth"] = df_feat["InvoiceDate"].dt.to_period("M").astype(str)
    df_feat["Date"] = df_feat["InvoiceDate"].dt.date
    df_feat["DayOfWeek"] = df_feat["InvoiceDate"].dt.day_name()
    df_feat["Hour"] = df_feat["InvoiceDate"].dt.hour
    if hasattr(df_feat["InvoiceDate"].dt, "day_of_week"):
        dow = df_feat["InvoiceDate"].dt.day_of_week
    else:
        dow = df_feat["InvoiceDate"].dt.dayofweek
    df_feat["IsWeekend"] = dow.isin([5, 6]).astype(int)

    # Ensure Line Revenue
    if "TotalAmount" not in df_feat.columns:
        df_feat["TotalAmount"] = (df_feat["Quantity"] * df_feat["UnitPrice"]).round(2)

    logger.debug(f"Added temporal transaction features to {len(df_feat)} records.")
    return df_feat
