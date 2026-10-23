"""Transparent Rule-Based Business Recommendation Engine.

Generates evidence-backed strategic recommendations derived directly from calculated
customer metrics, RFM segments, retention rates, and revenue forecasts.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import pandas as pd

from config.logging_config import get_logger

logger = get_logger("src.recommendations.rules")

@dataclass
class RecommendationItem:
    """Standardized recommendation item with observed evidence, rule, action, and limitation."""
    title: str
    category: str  # 'Retention', 'Concentration', 'Forecasting', 'Acquisition'
    priority: str  # 'High', 'Medium', 'Low'
    observed_evidence: str
    threshold_rule: str
    suggested_action: str
    limitation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def generate_recommendations(
    customer_metrics_df: pd.DataFrame,
    segment_summary_df: pd.DataFrame,
    repeat_kpis: Dict[str, float],
    forecast_results: Optional[Any] = None,
    clv_summary: Optional[Dict[str, float]] = None
) -> List[RecommendationItem]:
    """Evaluates computed analytical metrics against explicit business rules to produce actionable recommendations.

    Args:
        customer_metrics_df: Customer-level metrics.
        segment_summary_df: RFM segment summary table.
        repeat_kpis: Output of calculate_repeat_purchase_metrics.
        forecast_results: ForecastResults from time-series module.
        clv_summary: Output of summarize_clv_distribution.

    Returns:
        List of structured RecommendationItem instances.
    """
    recs: List[RecommendationItem] = []

    if customer_metrics_df.empty or segment_summary_df.empty:
        logger.warning("Insufficient data to generate business recommendations.")
        return recs

    total_customers = len(customer_metrics_df)
    total_revenue = customer_metrics_df["total_revenue"].sum()

    # Rule 1: High Customer Concentration / Key Account Vulnerability
    top_10_count = max(int(total_customers * 0.10), 1)
    top_10_rev = customer_metrics_df.nlargest(top_10_count, "total_revenue")["total_revenue"].sum()
    top_10_rev_share = (top_10_rev / total_revenue * 100.0) if total_revenue > 0 else 0.0

    if top_10_rev_share >= 50.0:
        recs.append(RecommendationItem(
            title="Mitigate High Revenue Concentration Risk",
            category="Concentration",
            priority="High",
            observed_evidence=f"The top 10% of customers ({top_10_count} accounts) generate {top_10_rev_share:.1f}% of total historical revenue (${top_10_rev:,.2f}).",
            threshold_rule="Top decile revenue concentration exceeds 50.0% threshold.",
            suggested_action="Establish VIP white-glove retention protocols and account health monitoring to protect key revenue contributors.",
            limitation="High concentration is standard in B2B/wholesale retail; aggressive diversification without protecting core accounts may diminish margins."
        ))

    # Rule 2: Single-Purchase Churn Vulnerability
    repeat_rate = repeat_kpis.get("Repeat Purchase Rate (%)", 0.0)
    single_count = repeat_kpis.get("Single-Purchase Customers", 0)
    single_share = (single_count / total_customers * 100.0) if total_customers > 0 else 0.0

    if single_share >= 40.0:
        recs.append(RecommendationItem(
            title="Accelerate Second-Purchase Conversion Sequence",
            category="Acquisition & Onboarding",
            priority="High",
            observed_evidence=f"{single_count} customers ({single_share:.1f}% of customer base) have made exactly one purchase and never returned.",
            threshold_rule="Single-purchase customer proportion exceeds 40.0% of total customer base.",
            suggested_action="Deploy an automated 14-day post-purchase email onboarding series offering complementary product recommendations and a time-bound incentive.",
            limitation="Some one-off buyers may be gift shoppers or non-recurring retail visitors with zero repeat propensity."
        ))

    # Rule 3: At-Risk High-Value Segment Dormancy
    at_risk_row = segment_summary_df[segment_summary_df["Segment"] == "At-Risk Customers"]
    if not at_risk_row.empty:
        at_risk_rev_share = float(at_risk_row["Revenue Share (%)"].iloc[0])
        at_risk_cust_count = int(at_risk_row["Customer Count"].iloc[0])
        if at_risk_rev_share >= 10.0 or at_risk_cust_count > 15:
            recs.append(RecommendationItem(
                title="Launch Win-Back Campaign for 'At-Risk Customers'",
                category="Retention",
                priority="High",
                observed_evidence=f"The 'At-Risk Customers' segment contains {at_risk_cust_count} accounts accounting for {at_risk_rev_share:.1f}% of total historical revenue.",
                threshold_rule="At-Risk segment revenue contribution exceeds 10.0% of cumulative sales.",
                suggested_action="Execute a personalized win-back campaign with reactivation discounts and survey inquiries to diagnose dropout reasons.",
                limitation="Win-back promotions carry margin degradation risk if applied to permanently churned accounts or bargain hunters."
            ))

    # Rule 4: Potential Loyalists Upsell Strategy
    pot_loyal_row = segment_summary_df[segment_summary_df["Segment"] == "Potential Loyalists"]
    if not pot_loyal_row.empty:
        pot_count = int(pot_loyal_row["Customer Count"].iloc[0])
        pot_share = float(pot_loyal_row["Customer Share (%)"].iloc[0])
        if pot_count >= 10:
            recs.append(RecommendationItem(
                title="Cultivate Potential Loyalists into Brand Advocates",
                category="Retention",
                priority="Medium",
                observed_evidence=f"{pot_count} customers ({pot_share:.1f}% of user base) fall into 'Potential Loyalists' with recent purchases and moderate spend.",
                threshold_rule="Potential Loyalists cluster exceeds 10 unique customers.",
                suggested_action="Introduce tiered loyalty reward thresholds and early access to new product releases to stimulate order frequency.",
                limitation="Does not measure incremental uplift directly; requires A/B test validation against an unexposed control group."
            ))

    # Rule 5: Monthly Revenue Trend / Forecast Trajectory
    if forecast_results and not forecast_results.monthly_history.empty and len(forecast_results.monthly_history) >= 3:
        monthly_revs = forecast_results.monthly_history["TotalRevenue"]
        last_m = monthly_revs.iloc[-1]
        prev_m = monthly_revs.iloc[-2]
        mom_change = ((last_m - prev_m) / prev_m * 100.0) if prev_m > 0 else 0.0

        if abs(mom_change) >= 20.0:
            direction = "surged" if mom_change > 0 else "declined"
            recs.append(RecommendationItem(
                title=f"Investigate Recent Month-over-Month Revenue Shift ({mom_change:+.1f}%)",
                category="Forecasting",
                priority="Medium",
                observed_evidence=f"Revenue {direction} from ${prev_m:,.2f} to ${last_m:,.2f} ({mom_change:+.1f}%) in the most recent closed month.",
                threshold_rule="Absolute month-over-month revenue velocity exceeds 20.0% threshold.",
                suggested_action="Perform category-level and geographic breakdown to verify whether change stems from seasonal promotions or structural churn.",
                limitation="Short-term monthly swings can be influenced by irregular billing cycles, inventory stockouts, or calendar effects."
            ))

    # Rule 6: Hibernating Segment Re-engagement vs Sunset Policy
    hib_row = segment_summary_df[segment_summary_df["Segment"] == "Hibernating Customers"]
    if not hib_row.empty:
        hib_count = int(hib_row["Customer Count"].iloc[0])
        hib_share = float(hib_row["Customer Share (%)"].iloc[0])
        if hib_share >= 25.0:
            recs.append(RecommendationItem(
                title="Audit Hibernating Customer Base for Contact List Hygiene",
                category="Retention",
                priority="Low",
                observed_evidence=f"{hib_count} customers ({hib_share:.1f}% of total) are classified as 'Hibernating' with low recency, frequency, and spend.",
                threshold_rule="Hibernating accounts represent 25.0% or more of customer directory.",
                suggested_action="Run a final low-cost re-engagement message, then sunset non-responders from active marketing automation to preserve domain deliverability.",
                limitation="Aggressive list pruning slightly decreases broad reach, though it protects sender reputation."
            ))

    logger.info(f"Generated {len(recs)} data-driven recommendations.")
    return recs
