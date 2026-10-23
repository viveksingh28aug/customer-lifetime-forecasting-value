"""RFM (Recency, Frequency, Monetary) Segmentation Module.

Implements robust quantile-based RFM scoring with tie handling,
deterministic segment classification, and segment-level revenue contribution analysis.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config.settings import RFM_SEGMENT_DEFINITIONS
from config.logging_config import get_logger

logger = get_logger("src.segmentation.rfm")

def compute_quantile_score(series: pd.Series, num_bins: int = 5, ascending: bool = True) -> pd.Series:
    """Computes quantile-based scores from 1 to num_bins, handling ties gracefully.

    Args:
        series: Numerical series to score.
        num_bins: Number of quantile bins (typically 4 or 5).
        ascending: True if higher value gets higher score; False if lower value gets higher score.

    Returns:
        Series of integer scores from 1 to num_bins.
    """
    if series.empty or len(series.dropna()) == 0:
        return pd.Series(dtype=int)

    # Use percent rank to avoid qcut duplicate bin edges failure on tied values
    percent_ranks = series.rank(pct=True, method="first")
    
    if not ascending:
        # Invert rank: lowest value gets highest rank (e.g., lower recency days is better)
        percent_ranks = 1.0 - percent_ranks

    # Scale to 1..num_bins
    scores = np.ceil(percent_ranks * num_bins).astype(int)
    # Clip between 1 and num_bins
    scores = scores.clip(lower=1, upper=num_bins)
    return scores

def classify_rfm_segment(r: int, f: int, m: int) -> str:
    """Classifies an RFM score tuple into documented business customer segments.

    Rules:
    - Champions: R >= 4 and F >= 4
    - Loyal Customers: R >= 3 and F >= 3
    - Potential Loyalists: R >= 4 and F < 3
    - Recent Customers: R >= 4 and F == 1
    - Customers Needing Attention: (R == 2 or R == 3) and F >= 2
    - At-Risk Customers: R <= 2 and F >= 3
    - Hibernating Customers: R <= 2 and F <= 2
    """
    if r >= 4 and f >= 4:
        return "Champions"
    elif r >= 4 and f == 1:
        return "Recent Customers"
    elif r >= 4 and f < 4:
        return "Potential Loyalists"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r <= 2 and f >= 3:
        return "At-Risk Customers"
    elif (r in [2, 3]) and f >= 2:
        return "Customers Needing Attention"
    else:
        return "Hibernating Customers"

def perform_rfm_segmentation(
    customer_metrics_df: pd.DataFrame,
    num_bins: int = 5,
    run_id: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes RFM scores, assigns segments, and summarizes segment performance.

    Args:
        customer_metrics_df: DataFrame output from compute_customer_metrics.
        num_bins: Number of score bins (default 5 for 1-5 scale).
        run_id: Pipeline execution identifier.

    Returns:
        Tuple of:
          - Enriched customer DataFrame with RFM scores and Segment
          - Segment summary DataFrame with counts, total revenue, and contribution percentages
    """
    if customer_metrics_df.empty:
        logger.warning("Empty customer metrics provided to RFM segmentation.")
        return pd.DataFrame(), pd.DataFrame()

    df_rfm = customer_metrics_df.copy()

    # Recency: lower days is better (ascending=False)
    df_rfm["recency_score"] = compute_quantile_score(
        df_rfm["recency_days"], num_bins=num_bins, ascending=False
    )

    # Frequency: higher invoices is better (ascending=True)
    df_rfm["frequency_score"] = compute_quantile_score(
        df_rfm["num_invoices"], num_bins=num_bins, ascending=True
    )

    # Monetary: higher revenue is better (ascending=True)
    df_rfm["monetary_score"] = compute_quantile_score(
        df_rfm["total_revenue"], num_bins=num_bins, ascending=True
    )

    # Combined composite score (e.g., '555')
    df_rfm["rfm_score"] = (
        df_rfm["recency_score"].astype(str)
        + df_rfm["frequency_score"].astype(str)
        + df_rfm["monetary_score"].astype(str)
    )

    # Segment classification
    df_rfm["segment"] = [
        classify_rfm_segment(r, f, m)
        for r, f, m in zip(df_rfm["recency_score"], df_rfm["frequency_score"], df_rfm["monetary_score"])
    ]

    if run_id:
        df_rfm["run_id"] = run_id
    df_rfm["assigned_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Segment Summary
    total_customers = len(df_rfm)
    total_revenue = df_rfm["total_revenue"].sum()

    summary_records = []
    for segment_name, description in RFM_SEGMENT_DEFINITIONS.items():
        seg_data = df_rfm[df_rfm["segment"] == segment_name]
        cust_count = len(seg_data)
        seg_revenue = seg_data["total_revenue"].sum() if cust_count > 0 else 0.0
        avg_aov = seg_data["avg_order_value"].mean() if cust_count > 0 else 0.0
        avg_recency = seg_data["recency_days"].mean() if cust_count > 0 else 0.0

        summary_records.append({
            "Segment": segment_name,
            "Customer Count": cust_count,
            "Customer Share (%)": round((cust_count / total_customers * 100), 2) if total_customers > 0 else 0.0,
            "Total Revenue": round(seg_revenue, 2),
            "Revenue Share (%)": round((seg_revenue / total_revenue * 100), 2) if total_revenue > 0 else 0.0,
            "Avg AOV": round(avg_aov, 2),
            "Avg Recency (Days)": round(avg_recency, 1),
            "Description": description,
        })

    segment_summary = pd.DataFrame(summary_records).sort_values(by="Total Revenue", ascending=False).reset_index(drop=True)

    logger.info(f"RFM segmentation completed across {total_customers} customers into {len(segment_summary)} segments.")
    return df_rfm, segment_summary
