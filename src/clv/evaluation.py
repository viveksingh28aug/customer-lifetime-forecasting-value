"""CLV Model Evaluation and Summary Metrics Module.

Computes distribution statistics, concentration metrics, and model comparison tables.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger("src.clv.evaluation")

def summarize_clv_distribution(clv_df: pd.DataFrame) -> Dict[str, float]:
    """Calculates key statistical summaries of customer lifetime value predictions.

    Args:
        clv_df: DataFrame containing 'predicted_clv' column.

    Returns:
        Dictionary of statistical summary metrics.
    """
    if clv_df.empty or "predicted_clv" not in clv_df.columns:
        return {}

    clv_vals = clv_df["predicted_clv"].dropna()
    if len(clv_vals) == 0:
        return {}

    total_pred = float(clv_vals.sum())
    top_10_pct_count = max(int(len(clv_vals) * 0.10), 1)
    top_10_pct_rev = float(clv_vals.nlargest(top_10_pct_count).sum())
    top_10_share = (top_10_pct_rev / total_pred * 100.0) if total_pred > 0 else 0.0

    return {
        "Total Predicted CLV": round(total_pred, 2),
        "Mean CLV": round(float(clv_vals.mean()), 2),
        "Median CLV": round(float(clv_vals.median()), 2),
        "Std CLV": round(float(clv_vals.std()), 2) if len(clv_vals) > 1 else 0.0,
        "Min CLV": round(float(clv_vals.min()), 2),
        "Max CLV": round(float(clv_vals.max()), 2),
        "Top 10% Customer Value Share (%)": round(top_10_share, 1),
        "Total Customers Evaluated": len(clv_vals),
    }

def compare_clv_models(model_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Produces a side-by-side comparison summary across multiple CLV approaches.

    Args:
        model_dfs: Dict of {Model Name: Prediction DataFrame}

    Returns:
        Comparison summary DataFrame.
    """
    records = []
    for name, df in model_dfs.items():
        if df.empty or "predicted_clv" not in df.columns:
            continue
        stats_dict = summarize_clv_distribution(df)
        records.append({
            "Model Name": name,
            "Customers": stats_dict.get("Total Customers Evaluated", 0),
            "Total Value": stats_dict.get("Total Predicted CLV", 0.0),
            "Mean CLV": stats_dict.get("Mean CLV", 0.0),
            "Median CLV": stats_dict.get("Median CLV", 0.0),
            "Max CLV": stats_dict.get("Max CLV", 0.0),
            "Top 10% Share (%)": stats_dict.get("Top 10% Customer Value Share (%)", 0.0),
        })

    return pd.DataFrame(records)
