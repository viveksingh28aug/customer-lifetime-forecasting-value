"""Customer Retention and Inactivity Analysis Module.

Analyzes repeat purchasing rates, customer inactivity distributions,
and flags high-value at-risk accounts using documented heuristic inactivity criteria.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger("src.retention.retention_analysis")

def calculate_repeat_purchase_metrics(customer_metrics_df: pd.DataFrame) -> Dict[str, float]:
    """Calculates overall repeat purchase KPIs across the customer base.

    Args:
        customer_metrics_df: Aggregated customer metrics DataFrame.

    Returns:
        Dictionary of repeat purchasing indicators.
    """
    if customer_metrics_df.empty:
        return {}

    total_customers = len(customer_metrics_df)
    repeat_customers = customer_metrics_df[customer_metrics_df["num_invoices"] > 1]
    num_repeat = len(repeat_customers)
    num_single = total_customers - num_repeat

    repeat_rate = (num_repeat / total_customers * 100.0) if total_customers > 0 else 0.0
    repeat_revenue = repeat_customers["total_revenue"].sum()
    total_revenue = customer_metrics_df["total_revenue"].sum()
    repeat_rev_share = (repeat_revenue / total_revenue * 100.0) if total_revenue > 0 else 0.0

    return {
        "Total Customers": total_customers,
        "Repeat Customers": num_repeat,
        "Single-Purchase Customers": num_single,
        "Repeat Purchase Rate (%)": round(repeat_rate, 2),
        "Repeat Customers Revenue Share (%)": round(repeat_rev_share, 2),
        "Avg Orders per Repeat Customer": round(float(repeat_customers["num_invoices"].mean()), 2) if num_repeat > 0 else 1.0,
        "Avg Spend Repeat Customer": round(float(repeat_customers["total_revenue"].mean()), 2) if num_repeat > 0 else 0.0,
        "Avg Spend Single Customer": round(float(customer_metrics_df[customer_metrics_df["num_invoices"] == 1]["total_revenue"].mean()), 2) if num_single > 0 else 0.0,
    }

def compute_inactivity_distribution(customer_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Categorizes customers into operational inactivity buckets based on recency days.

    Buckets:
    - Active (0 - 30 days)
    - Cooling Down (31 - 90 days)
    - Warm Inactive (91 - 180 days)
    - Dormant (181 - 365 days)
    - Lapsed (> 365 days)
    """
    if customer_metrics_df.empty or "recency_days" not in customer_metrics_df.columns:
        return pd.DataFrame()

    bins = [-1, 30, 90, 180, 365, float("inf")]
    labels = ["Active (0-30d)", "Cooling Down (31-90d)", "Warm Inactive (91-180d)", "Dormant (181-365d)", "Lapsed (>365d)"]

    df_copy = customer_metrics_df.copy()
    df_copy["Inactivity_Bucket"] = pd.cut(df_copy["recency_days"], bins=bins, labels=labels)

    grouped = df_copy.groupby("Inactivity_Bucket", observed=False).agg(
        Customer_Count=("CustomerID", "count"),
        Total_Revenue=("total_revenue", "sum"),
        Avg_Revenue=("total_revenue", "mean")
    ).reset_index()

    total_c = len(df_copy)
    total_r = df_copy["total_revenue"].sum()

    grouped["Customer_Share_Pct"] = (grouped["Customer_Count"] / total_c * 100.0).round(2)
    grouped["Revenue_Share_Pct"] = (grouped["Total_Revenue"] / total_r * 100.0).round(2)
    grouped["Total_Revenue"] = grouped["Total_Revenue"].round(2)
    grouped["Avg_Revenue"] = grouped["Avg_Revenue"].round(2)

    return grouped

def identify_at_risk_customers(
    customer_df: pd.DataFrame,
    inactivity_threshold_days: int = 90,
    top_n: int = 50
) -> pd.DataFrame:
    """Identifies high-value customers who have lapsed beyond the inactivity threshold.

    Note:
      This is a documented heuristic inactivity indicator based on observed recency
      and historical monetary spend, rather than a probabilistic churn classification model.
    """
    if customer_df.empty or "recency_days" not in customer_df.columns:
        return pd.DataFrame()

    df_copy = customer_df.copy()
    median_revenue = df_copy["total_revenue"].median()

    # High-value customers who are inactive
    at_risk_mask = (df_copy["recency_days"] >= inactivity_threshold_days) & (df_copy["total_revenue"] >= median_revenue)
    at_risk_df = df_copy[at_risk_mask].copy()

    # Sort by total revenue descending
    at_risk_df = at_risk_df.sort_values(by="total_revenue", ascending=False).head(top_n).reset_index(drop=True)

    # Add suggested retention action
    def get_action(row):
        if row["total_revenue"] > df_copy["total_revenue"].quantile(0.90):
            return "VIP Concierge: Personal outreach or exclusive VIP win-back offer"
        elif row["num_invoices"] > 3:
            return "Loyalty Re-engagement: Automated multi-step email sequence with discount"
        else:
            return "Win-Back Survey: Product feedback check-in with promotional incentive"

    at_risk_df["Suggested_Action"] = at_risk_df.apply(get_action, axis=1)
    at_risk_df["Indicator_Type"] = "Heuristic Recency Threshold"

    logger.info(f"Identified {len(at_risk_df)} high-value at-risk accounts (Threshold: {inactivity_threshold_days} days).")
    return at_risk_df
