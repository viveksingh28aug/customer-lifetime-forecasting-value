"""Automated Executive Business Report Generator.

Compiles calculated analytical findings, segmentation performance, CLV distributions,
revenue forecast projections, and strategic recommendations into structured Markdown reports.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from config.settings import REPORTS_DIR
from config.logging_config import get_logger

logger = get_logger("src.reporting.report_generator")

def generate_markdown_report(
    metadata_dict: Dict[str, Any],
    dq_report_dict: Dict[str, Any],
    customer_kpis: Dict[str, Any],
    segment_summary_df: pd.DataFrame,
    clv_summary: Dict[str, Any],
    forecast_metrics_df: pd.DataFrame,
    forecast_future_df: pd.DataFrame,
    recommendations: List[Any],
    run_id: str,
) -> Tuple[str, Path]:
    """Generates a complete executive briefing document based purely on computed outputs.

    Args:
        metadata_dict: Ingestion and pipeline metadata.
        dq_report_dict: Data quality audit metrics.
        customer_kpis: Executive KPI dictionary.
        segment_summary_df: RFM segment metrics.
        clv_summary: Lifetime value statistical summary.
        forecast_metrics_df: Time-series holdout evaluation metrics.
        forecast_future_df: Future forecast table.
        recommendations: List of RecommendationItem objects.
        run_id: Pipeline execution ID.

    Returns:
        Tuple of (Report Markdown String, Path to Saved Report File)
    """
    report_lines: List[str] = []

    report_lines.append("# Executive Analytics & Business Intelligence Report")
    report_lines.append(f"**Customer Lifetime Value (CLV) & Revenue Forecasting Engine**  ")
    report_lines.append(f"*Generated:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | *Run ID:* `{run_id}`  \n")
    report_lines.append("---")

    # 1. Dataset Provenance & Data Quality Health
    report_lines.append("## 1. Data Provenance & Ingestion Audit")
    source_name = metadata_dict.get("dataset_source", "Unknown")
    is_synth = metadata_dict.get("is_synthetic", False)
    synth_badge = "⚠️ **SYNTHETIC DATA ACTIVE**" if is_synth else "✅ **AUTHENTIC DATA SOURCE**"

    report_lines.append(f"- **Data Source:** {source_name} ({synth_badge})")
    report_lines.append(f"- **Total Raw Transactions:** {dq_report_dict.get('total_raw_rows', 0):,}")
    report_lines.append(f"- **Duplicates Excluded:** {dq_report_dict.get('duplicate_rows', 0):,}")
    report_lines.append(f"- **Missing Customer IDs Filtered:** {dq_report_dict.get('missing_customer_ids', 0):,}")
    report_lines.append(f"- **Returns / Cancellations Handled:** {dq_report_dict.get('return_records', 0):,}")
    report_lines.append(f"- **Final Cleaned Sales Transactions:** {dq_report_dict.get('final_cleaned_rows', 0):,}")
    report_lines.append(f"- **Data Quality Health Score:** {dq_report_dict.get('data_quality_score', 0):.1f}%\n")

    # 2. Key Executive Performance Indicators
    report_lines.append("## 2. Key Executive Performance Indicators (KPIs)")
    report_lines.append("| Metric | Value | Description |")
    report_lines.append("| :--- | :--- | :--- |")
    report_lines.append(f"| **Total Historical Revenue** | ${customer_kpis.get('total_revenue', 0.0):,.2f} | Cumulative sales from cleaned transactions |")
    report_lines.append(f"| **Unique Active Customers** | {customer_kpis.get('unique_customers', 0):,} | Distinct validated customer accounts |")
    report_lines.append(f"| **Total Invoices / Orders** | {customer_kpis.get('total_invoices', 0):,} | Total distinct purchase orders placed |")
    report_lines.append(f"| **Average Order Value (AOV)** | ${customer_kpis.get('avg_order_value', 0.0):,.2f} | Revenue generated per invoice |")
    report_lines.append(f"| **Average Customer Spend** | ${customer_kpis.get('avg_customer_revenue', 0.0):,.2f} | Cumulative revenue per customer |")
    report_lines.append(f"| **Repeat Customer Rate** | {customer_kpis.get('repeat_rate', 0.0):.1f}% | Percentage of customers with > 1 purchase |")
    report_lines.append(f"| **Repeat Revenue Contribution** | {customer_kpis.get('repeat_rev_share', 0.0):.1f}% | Share of total sales from repeat buyers |\n")

    # 3. RFM Customer Segmentation Summary
    report_lines.append("## 3. RFM Customer Segmentation Breakdown")
    if not segment_summary_df.empty:
        report_lines.append("| Segment | Customer Count | Cust Share (%) | Total Revenue | Rev Share (%) | Avg AOV | Avg Recency (d) |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for _, row in segment_summary_df.iterrows():
            report_lines.append(
                f"| **{row['Segment']}** | {row['Customer Count']:,} | {row['Customer Share (%)']:.1f}% | "
                f"${row['Total Revenue']:,.2f} | {row['Revenue Share (%)']:.1f}% | "
                f"${row['Avg AOV']:,.2f} | {row['Avg Recency (Days)']:.1f} |"
            )
        report_lines.append("")
    else:
        report_lines.append("*No RFM segmentation data available.*\n")

    # 4. Customer Lifetime Value (CLV) Outlook
    report_lines.append("## 4. Customer Lifetime Value (CLV) Outlook")
    if clv_summary:
        report_lines.append(f"- **Total Estimated Pipeline Value:** ${clv_summary.get('Total Predicted CLV', 0.0):,.2f}")
        report_lines.append(f"- **Mean Customer Lifetime Value:** ${clv_summary.get('Mean CLV', 0.0):,.2f}")
        report_lines.append(f"- **Median Customer Value:** ${clv_summary.get('Median CLV', 0.0):,.2f}")
        report_lines.append(f"- **Top 10% Customer Value Concentration:** {clv_summary.get('Top 10% Customer Value Share (%)', 0.0):.1f}%\n")
    else:
        report_lines.append("*No CLV summary metrics available.*\n")

    # 5. Revenue Forecast & Time-Series Evaluation
    report_lines.append("## 5. Revenue Forecast & Evaluation Summary")
    if not forecast_metrics_df.empty:
        report_lines.append("### Chronological Test Evaluation Metrics (Held-out Actuals)")
        report_lines.append("| Model | MAE | RMSE | WAPE (%) | MAPE (%) | Holdout Months |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for _, row in forecast_metrics_df.iterrows():
            mae_str = f"${row['MAE']:,.2f}" if pd.notna(row.get('MAE')) else "N/A"
            rmse_str = f"${row['RMSE']:,.2f}" if pd.notna(row.get('RMSE')) else "N/A"
            wape_str = f"{row['WAPE (%)']:.1f}%" if pd.notna(row.get('WAPE (%)')) else "N/A"
            mape_str = f"{row['MAPE (%)']:.1f}%" if pd.notna(row.get('MAPE (%)')) else "N/A"
            obs = row.get('Observations', 'N/A')
            report_lines.append(f"| **{row.get('Model', 'Unknown')}** | {mae_str} | {rmse_str} | {wape_str} | {mape_str} | {obs} |")
        report_lines.append("")

    if not forecast_future_df.empty:
        report_lines.append("### Future Revenue Forecast Projections")
        report_lines.append("| Period | Model | Projected Revenue | Lower Bound (95%) | Upper Bound (95%) |")
        report_lines.append("| :---: | :--- | :---: | :---: | :---: |")
        # Display up to 6 months of future projection
        for _, row in forecast_future_df.head(12).iterrows():
            report_lines.append(
                f"| {row['YearMonth']} | {row['ModelName']} | "
                f"${row['PredictedRevenue']:,.2f} | ${row['LowerBound']:,.2f} | ${row['UpperBound']:,.2f} |"
            )
        report_lines.append("")

    # 6. Strategic Recommendations
    report_lines.append("## 6. Strategic & Operational Recommendations")
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            r = rec.to_dict() if hasattr(rec, "to_dict") else rec
            report_lines.append(f"### {idx}. [{r.get('priority', 'Medium')} Priority] {r.get('title', 'Recommendation')}")
            report_lines.append(f"- **Category:** {r.get('category', 'General')}")
            report_lines.append(f"- **Observed Evidence:** {r.get('observed_evidence')}")
            report_lines.append(f"- **Trigger Rule:** {r.get('threshold_rule')}")
            report_lines.append(f"- **Suggested Action:** {r.get('suggested_action')}")
            report_lines.append(f"- **Strategic Limitation:** *{r.get('limitation')}*\n")
    else:
        report_lines.append("*No automated recommendations triggered under current thresholds.*\n")

    report_lines.append("---")
    report_lines.append("*End of Report. Generated automatically by the Customer Lifetime Value (CLV) & Revenue Forecasting Engine.*")

    full_markdown = "\n".join(report_lines)

    # Save to file
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"business_report_{run_id}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    logger.info(f"Executive markdown report saved to {report_path}")
    return full_markdown, report_path
