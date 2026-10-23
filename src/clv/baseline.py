"""Transparent Baseline Customer Lifetime Value (CLV) Models.

Implements interpretable, rule-based CLV estimation using observed customer purchasing
cadence, order values, configurable horizons, profit margins, and discount rates.
"""

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from config.settings import DEFAULT_CLV_HORIZON_MONTHS, DEFAULT_DISCOUNT_RATE, DEFAULT_GROSS_MARGIN
from config.logging_config import get_logger

logger = get_logger("src.clv.baseline")

def compute_historical_baseline_clv(
    customer_metrics_df: pd.DataFrame,
    gross_margin: float = DEFAULT_GROSS_MARGIN,
    horizon_months: int = DEFAULT_CLV_HORIZON_MONTHS,
    run_id: Optional[str] = None
) -> pd.DataFrame:
    """Estimates historical baseline customer value based on observed transaction history.

    Assumptions:
    - Customer historical revenue is annualized based on observed active lifespan (or base year).
    - Margin is applied to convert revenue to gross profit.
    """
    if customer_metrics_df.empty:
        return pd.DataFrame()

    df_clv = customer_metrics_df.copy()

    # Annualized historical revenue: (total_revenue / max(lifespan_days, 30)) * 365
    lifespan_days = np.maximum(df_clv["customer_lifespan_days"], 30.0)
    annualized_revenue = (df_clv["total_revenue"] / lifespan_days) * 365.0
    horizon_revenue = annualized_revenue * (horizon_months / 12.0)

    df_clv["model_type"] = "Historical Baseline"
    df_clv["horizon_months"] = horizon_months
    df_clv["predicted_transactions"] = (df_clv["purchase_frequency"] * horizon_months).round(1)
    df_clv["predicted_spend_per_tx"] = df_clv["avg_order_value"]
    df_clv["predicted_revenue"] = horizon_revenue.round(2)
    df_clv["predicted_clv"] = (horizon_revenue * gross_margin).round(2)
    df_clv["gross_margin"] = gross_margin
    df_clv["discount_rate"] = 0.0
    if run_id:
        df_clv["run_id"] = run_id
    df_clv["prediction_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return df_clv

def compute_transparent_forecast_clv(
    customer_metrics_df: pd.DataFrame,
    horizon_months: int = DEFAULT_CLV_HORIZON_MONTHS,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    gross_margin: float = DEFAULT_GROSS_MARGIN,
    recency_decay_factor: bool = True,
    run_id: Optional[str] = None
) -> pd.DataFrame:
    """Estimates future CLV using observed purchasing frequency, AOV, and recency decay.

    Formula:
      Active Probability Weight: exp(-recency_days / 180) (if recency_decay enabled)
      Expected Transactions = purchase_frequency_per_month * horizon_months * Active Weight
      Expected Revenue = Expected Transactions * avg_order_value
      Predicted CLV = (Expected Revenue * gross_margin) / (1 + discount_rate * (horizon_months / 12))

    Assumptions:
    - Average Order Value remains stable over the horizon.
    - Dormant customers with high recency days have lower probability of making purchases.
    - Discount rate is applied to present-value future cash flows.
    """
    if customer_metrics_df.empty:
        return pd.DataFrame()

    df_clv = customer_metrics_df.copy()

    # Recency decay weight (damping factor for inactive customers)
    # Customers active in last 30 days have weight ~0.85-1.0; dormant (>365 days) dampens towards 0
    if recency_decay_factor:
        activity_weight = np.exp(-df_clv["recency_days"] / 180.0)
    else:
        activity_weight = np.ones(len(df_clv))

    # Expected transactions over horizon
    pred_tx = (df_clv["purchase_frequency"] * horizon_months * activity_weight).round(2)
    pred_tx = np.maximum(pred_tx, 0.0)

    # Expected future revenue
    expected_future_rev = (pred_tx * df_clv["avg_order_value"]).round(2)

    # Discount factor
    discount_factor = 1.0 + (discount_rate * (horizon_months / 12.0))

    # Net present CLV
    predicted_clv = ((expected_future_rev * gross_margin) / discount_factor).round(2)

    result_df = pd.DataFrame({
        "CustomerID": df_clv["CustomerID"],
        "model_type": "Transparent Forecast",
        "horizon_months": horizon_months,
        "predicted_transactions": pred_tx,
        "predicted_spend_per_tx": df_clv["avg_order_value"],
        "predicted_revenue": expected_future_rev,
        "predicted_clv": predicted_clv,
        "gross_margin": gross_margin,
        "discount_rate": discount_rate,
        "recency_days": df_clv["recency_days"],
        "total_historical_revenue": df_clv["total_revenue"],
        "num_invoices": df_clv["num_invoices"],
    })

    if "segment" in df_clv.columns:
        result_df["segment"] = df_clv["segment"]

    if run_id:
        result_df["run_id"] = run_id
    result_df["prediction_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(
        f"Computed Transparent Forecast CLV for {len(result_df)} customers. "
        f"Mean CLV: ${predicted_clv.mean():.2f}, Horizon: {horizon_months}M."
    )
    return result_df
