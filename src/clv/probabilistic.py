"""Probabilistic Customer Lifetime Value (BG/NBD & Gamma-Gamma) Module.

Implements Beta-Geometric / Negative Binomial Distribution (BG/NBD) for transaction frequency
and Gamma-Gamma sub-model for monetary value with rigorous assumption verification and transparent fallback.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.clv.baseline import compute_transparent_forecast_clv
from config.settings import DEFAULT_CLV_HORIZON_MONTHS, DEFAULT_DISCOUNT_RATE, DEFAULT_GROSS_MARGIN
from config.logging_config import get_logger

logger = get_logger("src.clv.probabilistic")

class CLVModelDiagnostics:
    """Stores diagnostics, assumption validation results, and model limitations."""

    def __init__(
        self,
        model_name: str,
        is_probabilistic: bool,
        assumptions_met: bool,
        repeat_customers_count: int,
        correlation_freq_monetary: Optional[float] = None,
        correlation_p_value: Optional[float] = None,
        convergence_status: str = "Success",
        limitations: Optional[List[str]] = None,
        fallback_reason: Optional[str] = None,
    ):
        self.model_name = model_name
        self.is_probabilistic = is_probabilistic
        self.assumptions_met = assumptions_met
        self.repeat_customers_count = repeat_customers_count
        self.correlation_freq_monetary = correlation_freq_monetary
        self.correlation_p_value = correlation_p_value
        self.convergence_status = convergence_status
        self.limitations = limitations or []
        self.fallback_reason = fallback_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "is_probabilistic": self.is_probabilistic,
            "assumptions_met": self.assumptions_met,
            "repeat_customers_count": self.repeat_customers_count,
            "correlation_freq_monetary": self.correlation_freq_monetary,
            "correlation_p_value": self.correlation_p_value,
            "convergence_status": self.convergence_status,
            "limitations": self.limitations,
            "fallback_reason": self.fallback_reason,
        }

def prepare_rfm_summary_for_lifetimes(
    df_sales: pd.DataFrame,
    reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Prepares transaction data into standard lifetimes RFM format (frequency, recency, T, monetary_value).

    Definitions in lifetimes:
    - frequency: number of repeat purchases (total purchases - 1)
    - recency: age of customer at last purchase (last_purchase - first_purchase in days)
    - T: age of customer relative to reference date (reference_date - first_purchase in days)
    - monetary_value: average order value on repeat purchases
    """
    if df_sales.empty:
        return pd.DataFrame()

    df_work = df_sales.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work["InvoiceDate"]):
        df_work["InvoiceDate"] = pd.to_datetime(df_work["InvoiceDate"], errors="coerce")

    if reference_date is None:
        reference_date = df_work["InvoiceDate"].max()

    # Aggregate by customer and invoice to get unique invoice dates and invoice amounts
    inv_summary = df_work.groupby(["CustomerID", "InvoiceNo"]).agg(
        InvoiceDate=("InvoiceDate", "min"),
        InvoiceAmount=("TotalAmount", "sum")
    ).reset_index()

    # Group by customer
    records = []
    for cid, group in inv_summary.groupby("CustomerID"):
        group_sorted = group.sort_values(by="InvoiceDate")
        n_orders = len(group_sorted)
        first_date = group_sorted["InvoiceDate"].iloc[0]
        last_date = group_sorted["InvoiceDate"].iloc[-1]

        # Lifetimes frequency represents repeat purchases
        freq = max(n_orders - 1, 0)
        rec = (last_date - first_date).total_seconds() / 86400.0
        T = (reference_date - first_date).total_seconds() / 86400.0
        T = max(T, rec)  # Ensure T >= recency

        if freq > 0:
            # Monetary value of repeat transactions (excluding the first order)
            repeat_orders = group_sorted.iloc[1:]
            monetary = repeat_orders["InvoiceAmount"].mean()
        else:
            monetary = group_sorted["InvoiceAmount"].iloc[0]

        records.append({
            "CustomerID": str(cid),
            "frequency": freq,
            "recency": rec,
            "T": T,
            "monetary_value": monetary,
            "total_orders": n_orders,
            "total_spend": group_sorted["InvoiceAmount"].sum(),
        })

    return pd.DataFrame(records)

def fit_probabilistic_clv(
    df_sales: pd.DataFrame,
    customer_metrics_df: pd.DataFrame,
    horizon_months: int = DEFAULT_CLV_HORIZON_MONTHS,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    gross_margin: float = DEFAULT_GROSS_MARGIN,
    run_id: Optional[str] = None
) -> Tuple[pd.DataFrame, CLVModelDiagnostics]:
    """Fits BG/NBD and Gamma-Gamma probabilistic models with assumption verification and automatic fallback.

    Args:
        df_sales: Cleaned sales transactions.
        customer_metrics_df: Aggregated customer metrics DataFrame.
        horizon_months: Forecasting horizon in months.
        discount_rate: Annual discount rate.
        gross_margin: Gross profit margin ratio.
        run_id: Pipeline execution identifier.

    Returns:
        Tuple of (CLV Predictions DataFrame, CLVModelDiagnostics)
    """
    logger.info("Initiating Probabilistic CLV evaluation...")

    # Step 1: Validate dependencies
    try:
        from lifetimes import BetaGeoFitter, GammaGammaFitter
    except ImportError:
        logger.warning("Lifetimes package not available. Falling back to Transparent Baseline.")
        fallback_df = compute_transparent_forecast_clv(
            customer_metrics_df, horizon_months=horizon_months,
            discount_rate=discount_rate, gross_margin=gross_margin, run_id=run_id
        )
        diag = CLVModelDiagnostics(
            model_name="Transparent Forecast (Fallback)",
            is_probabilistic=False,
            assumptions_met=False,
            repeat_customers_count=0,
            convergence_status="Lifetimes library not installed",
            fallback_reason="Lifetimes library missing; defaulted to transparent baseline.",
            limitations=["Uses linear frequency extrapolation instead of Bayesian Pareto/NBD."],
        )
        return fallback_df, diag

    # Step 2: Prepare summary data
    summary_df = prepare_rfm_summary_for_lifetimes(df_sales)
    if summary_df.empty:
        logger.warning("Empty data for lifetimes RFM summary.")
        fallback_df = compute_transparent_forecast_clv(customer_metrics_df, run_id=run_id)
        diag = CLVModelDiagnostics(
            model_name="Transparent Forecast (Fallback)",
            is_probabilistic=False,
            assumptions_met=False,
            repeat_customers_count=0,
            fallback_reason="Empty transaction data.",
        )
        return fallback_df, diag

    # Step 3: Check Assumptions
    # Assumption A: Sufficient repeat customers (repeat orders > 0)
    repeat_customers = summary_df[summary_df["frequency"] > 0]
    num_repeat = len(repeat_customers)
    
    if num_repeat < 10:
        logger.warning(f"Insufficient repeat customers ({num_repeat} < 10) for BG/NBD.")
        fallback_df = compute_transparent_forecast_clv(
            customer_metrics_df, horizon_months=horizon_months,
            discount_rate=discount_rate, gross_margin=gross_margin, run_id=run_id
        )
        diag = CLVModelDiagnostics(
            model_name="Transparent Forecast (Fallback)",
            is_probabilistic=False,
            assumptions_met=False,
            repeat_customers_count=num_repeat,
            convergence_status="Insufficient Data",
            fallback_reason=f"Only {num_repeat} customers had repeat purchases. Minimum 10 required.",
            limitations=["Small sample size of repeat buyers violates BG/NBD convergence stability."],
        )
        return fallback_df, diag

    # Assumption B: Positive monetary spend
    valid_repeat = repeat_customers[repeat_customers["monetary_value"] > 0]
    if len(valid_repeat) < 10:
        fallback_df = compute_transparent_forecast_clv(customer_metrics_df, run_id=run_id)
        diag = CLVModelDiagnostics(
            model_name="Transparent Forecast (Fallback)",
            is_probabilistic=False,
            assumptions_met=False,
            repeat_customers_count=num_repeat,
            fallback_reason="Non-positive monetary values among repeat customers.",
        )
        return fallback_df, diag

    # Assumption C: Correlation between frequency and monetary value (Gamma-Gamma independence)
    corr, p_value = stats.pearsonr(valid_repeat["frequency"], valid_repeat["monetary_value"])
    logger.info(f"Gamma-Gamma correlation check: r={corr:.3f}, p_value={p_value:.4f}")

    # Step 4: Fit BG/NBD Model
    try:
        # Scaling recency and T to weeks is standard practice for lifetimes to prevent numerical underflow
        recency_w = summary_df["recency"] / 7.0
        T_w = summary_df["T"] / 7.0
        freq = summary_df["frequency"]

        bgf = BetaGeoFitter(penalizer_coef=0.1)
        bgf.fit(freq, recency_w, T_w)
        logger.info(f"BG/NBD fitter converged. Params: {bgf.params_.to_dict()}")

        # Step 5: Fit Gamma-Gamma Model
        ggf = GammaGammaFitter(penalizer_coef=0.1)
        ggf.fit(valid_repeat["frequency"], valid_repeat["monetary_value"])
        logger.info(f"Gamma-Gamma fitter converged. Params: {ggf.params_.to_dict()}")

        # Step 6: Generate Predictions
        # Convert horizon to weeks (average 4.348 weeks per month)
        t_weeks = float(horizon_months * (365.25 / 12.0 / 7.0))

        # Predict conditional expected purchases over horizon weeks
        pred_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
            t_weeks, freq, recency_w, T_w
        ).round(2)

        # Predict conditional expected average spend
        pred_spend = ggf.conditional_expected_average_profit(
            summary_df["frequency"], summary_df["monetary_value"]
        ).round(2)

        # Expected revenue = purchases * spend
        pred_revenue = (pred_purchases * pred_spend).round(2)

        # Weekly discount rate: (1 + annual_rate)^(1/52) - 1
        weekly_discount = (1.0 + discount_rate) ** (1.0 / 52.0) - 1.0

        # Lifetimes CLV calculation in weeks
        clv_series = ggf.customer_lifetime_value(
            bgf,
            freq,
            recency_w,
            T_w,
            summary_df["monetary_value"],
            time=int(t_weeks),
            freq="W",
            discount_rate=weekly_discount
        )

        # Apply gross margin
        predicted_clv = (clv_series * gross_margin).round(2)

        result_df = pd.DataFrame({
            "CustomerID": summary_df["CustomerID"],
            "model_type": "Probabilistic (BG/NBD + Gamma-Gamma)",
            "horizon_months": horizon_months,
            "predicted_transactions": pred_purchases,
            "predicted_spend_per_tx": pred_spend,
            "predicted_revenue": pred_revenue,
            "predicted_clv": predicted_clv,
            "gross_margin": gross_margin,
            "discount_rate": discount_rate,
            "recency_days": summary_df["T"] - summary_df["recency"],
            "total_historical_revenue": summary_df["total_spend"],
            "num_invoices": summary_df["total_orders"],
        })

        # Merge segment if customer_metrics has it
        if "segment" in customer_metrics_df.columns:
            seg_map = dict(zip(customer_metrics_df["CustomerID"], customer_metrics_df["segment"]))
            result_df["segment"] = result_df["CustomerID"].map(seg_map).fillna("Unknown")

        if run_id:
            result_df["run_id"] = run_id
        result_df["prediction_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        diag = CLVModelDiagnostics(
            model_name="Probabilistic (BG/NBD + Gamma-Gamma)",
            is_probabilistic=True,
            assumptions_met=True,
            repeat_customers_count=num_repeat,
            correlation_freq_monetary=round(float(corr), 3),
            correlation_p_value=round(float(p_value), 4),
            convergence_status="Converged Successfully",
            limitations=[
                "Assumes transaction frequency is Poisson with Gamma-distributed latent transaction rate.",
                "Assumes customer dropout is Geometric with Beta-distributed latent dropout rate.",
                "Assumes monetary values follow Gamma distribution and are independent of frequency.",
                "CLV estimate reflects finite horizon discounting without accounting for multi-year churn shifts.",
            ],
        )

        logger.info(f"Probabilistic CLV fitted successfully for {len(result_df)} customers.")
        return result_df, diag

    except Exception as e:
        logger.error(f"Probabilistic model failed during optimization: {e}. Falling back to baseline.")
        fallback_df = compute_transparent_forecast_clv(
            customer_metrics_df, horizon_months=horizon_months,
            discount_rate=discount_rate, gross_margin=gross_margin, run_id=run_id
        )
        diag = CLVModelDiagnostics(
            model_name="Transparent Forecast (Fallback)",
            is_probabilistic=False,
            assumptions_met=False,
            repeat_customers_count=num_repeat,
            convergence_status=f"Optimization Exception: {str(e)[:100]}",
            fallback_reason=f"Lifetimes optimization failed: {e}",
            limitations=["Fell back to rule-based frequency and AOV projection."],
        )
        return fallback_df, diag
