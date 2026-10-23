"""End-to-End Pipeline Orchestrator Module.

Coordinates data ingestion, schema validation, data cleaning, feature extraction,
RFM segmentation, CLV modeling, revenue forecasting, database persistence, and reporting.
"""

from datetime import datetime
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
from pathlib import Path
import pandas as pd

from config.settings import DEFAULT_CLV_HORIZON_MONTHS, DEFAULT_DISCOUNT_RATE, DEFAULT_GROSS_MARGIN
from config.logging_config import get_logger
from database.db_manager import DatabaseManager
from src.ingestion.loader import load_dataset, find_local_dataset, DatasetMetadata
from src.ingestion.schema_validator import validate_and_map_schema
from src.preprocessing.cleaner import clean_retail_data
from src.preprocessing.data_quality import DataQualityReport
from src.features.transaction_features import add_transaction_features
from src.features.customer_features import compute_customer_metrics
from src.features.cohort_features import create_cohort_matrix
from src.segmentation.rfm import perform_rfm_segmentation
from src.clv.baseline import compute_historical_baseline_clv, compute_transparent_forecast_clv
from src.clv.probabilistic import fit_probabilistic_clv, CLVModelDiagnostics
from src.clv.evaluation import summarize_clv_distribution
from src.forecasting.revenue_forecast import run_revenue_forecast_pipeline, ForecastResults
from src.retention.retention_analysis import (
    calculate_repeat_purchase_metrics,
    compute_inactivity_distribution,
    identify_at_risk_customers,
)
from src.recommendations.rules import generate_recommendations, RecommendationItem
from src.reporting.report_generator import generate_markdown_report

logger = get_logger("src.pipeline.orchestrator")

class PipelineExecutionResult:
    """Encapsulates the complete state and outputs of an end-to-end pipeline run."""

    def __init__(
        self,
        run_id: str,
        status: str,
        metadata: DatasetMetadata,
        dq_report: DataQualityReport,
        raw_df: pd.DataFrame,
        cleaned_sales_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        customer_metrics_df: pd.DataFrame,
        segment_summary_df: pd.DataFrame,
        clv_predictions_df: pd.DataFrame,
        clv_diagnostics: CLVModelDiagnostics,
        forecast_results: ForecastResults,
        repeat_kpis: Dict[str, float],
        inactivity_dist_df: pd.DataFrame,
        at_risk_df: pd.DataFrame,
        cohort_counts_df: pd.DataFrame,
        cohort_retention_df: pd.DataFrame,
        recommendations: list,
        report_markdown: str,
        report_path: Path,
        duration_seconds: float,
        error_message: Optional[str] = None,
    ):
        self.run_id = run_id
        self.status = status
        self.metadata = metadata
        self.dq_report = dq_report
        self.raw_df = raw_df
        self.cleaned_sales_df = cleaned_sales_df
        self.returns_df = returns_df
        self.customer_metrics_df = customer_metrics_df
        self.segment_summary_df = segment_summary_df
        self.clv_predictions_df = clv_predictions_df
        self.clv_diagnostics = clv_diagnostics
        self.forecast_results = forecast_results
        self.repeat_kpis = repeat_kpis
        self.inactivity_dist_df = inactivity_dist_df
        self.at_risk_df = at_risk_df
        self.cohort_counts_df = cohort_counts_df
        self.cohort_retention_df = cohort_retention_df
        self.recommendations = recommendations
        self.report_markdown = report_markdown
        self.report_path = report_path
        self.duration_seconds = duration_seconds
        self.error_message = error_message

def run_pipeline(
    source: Optional[Union[str, Path, Any]] = None,
    file_type: Optional[str] = None,
    is_synthetic: bool = False,
    source_name: Optional[str] = None,
    custom_column_mapping: Optional[Dict[str, str]] = None,
    clv_horizon_months: int = DEFAULT_CLV_HORIZON_MONTHS,
    forecast_horizon_months: int = 6,
    gross_margin: float = DEFAULT_GROSS_MARGIN,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    db_manager: Optional[DatabaseManager] = None,
    persist_to_db: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> PipelineExecutionResult:
    """Executes the full modular analytics pipeline end-to-end.

    Args:
        source: Optional file path or buffer. If None, scans data/raw and data/synthetic.
        file_type: 'csv' or 'excel'.
        is_synthetic: Flag indicating whether input is synthetic.
        source_name: Display label for data source.
        custom_column_mapping: Column mapping dictionary if user specified.
        clv_horizon_months: CLV projection horizon (3, 6, 12).
        forecast_horizon_months: Revenue forecast horizon (3, 6, 12).
        gross_margin: Configurable gross profit margin (0.0 to 1.0).
        discount_rate: Annual discount rate.
        db_manager: DatabaseManager instance.
        persist_to_db: Whether to write outputs to DuckDB/SQLite.
        progress_callback: Function accepting (progress_float, status_text).

    Returns:
        PipelineExecutionResult containing all computed artifacts.
    """
    start_time = time.time()
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"=== Starting CLV & Revenue Forecasting Pipeline Run: {run_id} ===")

    def update_progress(pct: float, msg: str):
        logger.info(f"[{int(pct * 100)}%] {msg}")
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception as pe:
                logger.debug(f"Progress callback exception ignored: {pe}")

    update_progress(0.05, "Locating and verifying transaction data source...")

    # 1. Dataset Resolution & Ingestion
    if source is None:
        local_path, found_synth = find_local_dataset()
        if local_path is None:
            # Need to generate synthetic data if no data found
            from scripts.generate_synthetic_data import generate_synthetic_retail_data
            logger.info("No local dataset found. Auto-generating synthetic dataset fallback...")
            generate_synthetic_retail_data(seed=42)
            local_path, found_synth = find_local_dataset()

        source = local_path
        is_synthetic = found_synth
        source_name = "Synthetic Retail Dataset (Auto-Generated)" if found_synth else local_path.name

    raw_df, metadata = load_dataset(
        source=source,
        file_type=file_type,
        is_synthetic=is_synthetic,
        source_name=source_name,
    )

    update_progress(0.15, "Validating schema against retail standard columns...")
    # 2. Schema Validation & Mapping
    schema_res = validate_and_map_schema(raw_df, custom_mapping=custom_column_mapping)
    if not schema_res.is_valid:
        err = f"Schema validation failed: Missing columns {schema_res.missing_columns}"
        logger.error(err)
        raise ValueError(err)

    df_normalized = schema_res.normalized_df

    update_progress(0.25, "Cleaning transactions and generating Data Quality report...")
    # 3. Data Cleaning & DQ Audit
    cleaned_sales_df, returns_df, dq_report = clean_retail_data(
        df_normalized,
        run_id=run_id,
        drop_duplicates=True,
        drop_missing_customer=True,
        drop_cancellations=True,
        drop_non_positive_price=True,
    )

    if cleaned_sales_df.empty:
        raise ValueError("Cleaning produced 0 valid sales rows. Pipeline cannot continue.")

    update_progress(0.35, "Extracting transaction and customer behavioral features...")
    # 4. Feature Engineering
    cleaned_sales_df = add_transaction_features(cleaned_sales_df)
    customer_metrics_df = compute_customer_metrics(cleaned_sales_df, run_id=run_id)

    update_progress(0.45, "Executing RFM customer segmentation...")
    # 5. RFM Segmentation
    customer_metrics_df, segment_summary_df = perform_rfm_segmentation(customer_metrics_df, run_id=run_id)

    update_progress(0.55, "Fitting Customer Lifetime Value (CLV) models...")
    # 6. CLV Modeling (Probabilistic with BG/NBD & Gamma-Gamma; fallback if assumptions fail)
    clv_predictions_df, clv_diagnostics = fit_probabilistic_clv(
        df_sales=cleaned_sales_df,
        customer_metrics_df=customer_metrics_df,
        horizon_months=clv_horizon_months,
        discount_rate=discount_rate,
        gross_margin=gross_margin,
        run_id=run_id,
    )

    # Merge CLV prediction columns back into customer_metrics_df for unified table view
    if not clv_predictions_df.empty and "predicted_clv" in clv_predictions_df.columns:
        clv_subset = clv_predictions_df[["CustomerID", "predicted_clv", "predicted_transactions", "predicted_revenue"]].rename(
            columns={
                "predicted_clv": "predicted_clv",
                "predicted_transactions": "estimated_future_frequency",
                "predicted_revenue": "estimated_future_revenue",
            }
        )
        # Avoid duplicate columns on merge
        customer_metrics_df = customer_metrics_df.drop(
            columns=["predicted_clv", "estimated_future_frequency", "estimated_future_revenue"],
            errors="ignore"
        ).merge(clv_subset, on="CustomerID", how="left")

    update_progress(0.68, "Executing monthly revenue time-series forecasting...")
    # 7. Revenue Forecasting with Chronological Evaluation
    forecast_results = run_revenue_forecast_pipeline(
        df_sales=cleaned_sales_df,
        horizon_months=forecast_horizon_months,
        run_id=run_id,
    )

    update_progress(0.80, "Analyzing retention rates and customer cohorts...")
    # 8. Retention & Inactivity Analysis
    repeat_kpis = calculate_repeat_purchase_metrics(customer_metrics_df)
    inactivity_dist_df = compute_inactivity_distribution(customer_metrics_df)
    at_risk_df = identify_at_risk_customers(customer_metrics_df, inactivity_threshold_days=90)
    cohort_counts_df, cohort_retention_df = create_cohort_matrix(cleaned_sales_df)

    update_progress(0.88, "Evaluating business recommendation rules...")
    # 9. Business Recommendations
    clv_summary = summarize_clv_distribution(clv_predictions_df)
    recommendations = generate_recommendations(
        customer_metrics_df=customer_metrics_df,
        segment_summary_df=segment_summary_df,
        repeat_kpis=repeat_kpis,
        forecast_results=forecast_results,
        clv_summary=clv_summary,
    )

    update_progress(0.92, "Persisting analytical outputs to analytical database...")
    # 10. Database Persistence
    db = db_manager or DatabaseManager()
    db.init_schema()

    if persist_to_db:
        try:
            # Metadata table record
            meta_record = pd.DataFrame([{
                "run_id": run_id,
                "execution_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_source": metadata.source_name,
                "raw_row_count": len(raw_df),
                "cleaned_row_count": len(cleaned_sales_df),
                "unique_customers": len(customer_metrics_df),
                "status": "SUCCESS",
                "execution_duration_sec": round(time.time() - start_time, 2),
                "notes": f"Synthetic={metadata.is_synthetic}; CLV Model={clv_diagnostics.model_name}",
            }])
            db.save_dataframe("pipeline_metadata", meta_record, if_exists="append")

            # Data Quality record
            dq_record = pd.DataFrame([dq_report.to_sql_record()])
            db.save_dataframe("data_quality_reports", dq_record, if_exists="append")

            # Analytical DataFrames (saved or updated)
            # Store customer metrics & RFM
            db.save_dataframe("customer_metrics", customer_metrics_df, if_exists="replace")
            db.save_dataframe("rfm_segments", segment_summary_df, if_exists="replace")
            db.save_dataframe("clv_predictions", clv_predictions_df, if_exists="replace")

            if not forecast_results.future_forecasts.empty:
                db.save_dataframe("revenue_forecasts", forecast_results.future_forecasts, if_exists="replace")

            logger.info("Successfully persisted all analytical outputs to database.")
        except Exception as dbe:
            logger.error(f"Error persisting to analytical database: {dbe}")

    update_progress(0.97, "Generating automated executive report...")
    # 11. Executive Report Generation
    customer_kpis = {
        "total_revenue": customer_metrics_df["total_revenue"].sum(),
        "unique_customers": len(customer_metrics_df),
        "total_invoices": cleaned_sales_df["InvoiceNo"].nunique(),
        "avg_order_value": (customer_metrics_df["total_revenue"].sum() / cleaned_sales_df["InvoiceNo"].nunique()) if cleaned_sales_df["InvoiceNo"].nunique() > 0 else 0.0,
        "avg_customer_revenue": customer_metrics_df["total_revenue"].mean(),
        "repeat_rate": repeat_kpis.get("Repeat Purchase Rate (%)", 0.0),
        "repeat_rev_share": repeat_kpis.get("Repeat Customers Revenue Share (%)", 0.0),
    }

    report_md, report_path = generate_markdown_report(
        metadata_dict=metadata.to_dict(),
        dq_report_dict=dq_report.to_dict(),
        customer_kpis=customer_kpis,
        segment_summary_df=segment_summary_df,
        clv_summary=clv_summary,
        forecast_metrics_df=forecast_results.metrics_summary,
        forecast_future_df=forecast_results.future_forecasts,
        recommendations=recommendations,
        run_id=run_id,
    )

    duration = round(time.time() - start_time, 2)
    update_progress(1.0, f"Pipeline execution completed successfully in {duration}s.")

    return PipelineExecutionResult(
        run_id=run_id,
        status="SUCCESS",
        metadata=metadata,
        dq_report=dq_report,
        raw_df=raw_df,
        cleaned_sales_df=cleaned_sales_df,
        returns_df=returns_df,
        customer_metrics_df=customer_metrics_df,
        segment_summary_df=segment_summary_df,
        clv_predictions_df=clv_predictions_df,
        clv_diagnostics=clv_diagnostics,
        forecast_results=forecast_results,
        repeat_kpis=repeat_kpis,
        inactivity_dist_df=inactivity_dist_df,
        at_risk_df=at_risk_df,
        cohort_counts_df=cohort_counts_df,
        cohort_retention_df=cohort_retention_df,
        recommendations=recommendations,
        report_markdown=report_md,
        report_path=report_path,
        duration_seconds=duration,
    )
