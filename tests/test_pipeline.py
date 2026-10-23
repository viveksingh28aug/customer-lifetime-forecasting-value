"""End-to-End Analytics Pipeline Integration Tests."""

from pathlib import Path
import pandas as pd
import pytest

from src.pipeline.orchestrator import run_pipeline
from src.recommendations.rules import generate_recommendations
from database.db_manager import DatabaseManager

def test_pipeline_end_to_end_execution(sample_clean_transactions, tmp_path):
    """Test full pipeline runs from ingestion to DB persistence and reporting without error."""
    csv_path = tmp_path / "fixture_transactions.csv"
    sample_clean_transactions.to_csv(csv_path, index=False)

    db_path = tmp_path / "pipeline_test.duckdb"
    db = DatabaseManager(backend="duckdb", db_path=str(db_path))

    result = run_pipeline(
        source=csv_path,
        is_synthetic=True,
        source_name="Fixture Test",
        clv_horizon_months=12,
        forecast_horizon_months=3,
        db_manager=db,
        persist_to_db=True,
    )

    assert result.status == "SUCCESS"
    assert len(result.customer_metrics_df) == 4
    assert not result.segment_summary_df.empty
    assert not result.clv_predictions_df.empty
    assert Path(result.report_path).exists()
    assert result.duration_seconds > 0

    # Verify DuckDB records
    saved_custs = db.query("SELECT * FROM customer_metrics")
    assert len(saved_custs) == 4

    db.close()

def test_recommendation_rules_structure(sample_clean_transactions):
    """Test business recommendation items contain all 4 mandatory components."""
    from src.preprocessing.cleaner import clean_retail_data
    from src.features.customer_features import compute_customer_metrics
    from src.segmentation.rfm import perform_rfm_segmentation
    from src.retention.retention_analysis import calculate_repeat_purchase_metrics

    sales_df, _, _ = clean_retail_data(sample_clean_transactions)
    cust_metrics = compute_customer_metrics(sales_df)
    cust_metrics, seg_summary = perform_rfm_segmentation(cust_metrics)
    repeat_kpis = calculate_repeat_purchase_metrics(cust_metrics)

    recs = generate_recommendations(cust_metrics, seg_summary, repeat_kpis)
    assert len(recs) > 0

    for r in recs:
        assert r.observed_evidence != ""
        assert r.threshold_rule != ""
        assert r.suggested_action != ""
        assert r.limitation != ""
        assert r.priority in ["High", "Medium", "Low"]
