"""CLI Execution Entrypoint for the CLV & Revenue Forecasting Pipeline.

Allows executing the end-to-end pipeline from the terminal, with optional dataset
specification, forecasting horizons, and database selection.
"""

import argparse
from pathlib import Path
import sys

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.orchestrator import run_pipeline
from database.db_manager import DatabaseManager
from config.logging_config import get_logger

logger = get_logger("scripts.run_pipeline")

def main():
    parser = argparse.ArgumentParser(description="Run the CLV & Revenue Forecasting Analytics Pipeline.")
    parser.add_argument("--source", type=str, default=None, help="Path to CSV or Excel dataset")
    parser.add_argument("--db", type=str, default="duckdb", choices=["duckdb", "sqlite"], help="Database backend")
    parser.add_argument("--clv-horizon", type=int, default=12, help="CLV forecast horizon in months (3, 6, 12)")
    parser.add_argument("--forecast-horizon", type=int, default=6, help="Revenue forecast horizon in months (3, 6, 12)")
    parser.add_argument("--margin", type=float, default=1.0, help="Gross profit margin ratio (e.g. 0.40 for 40%%)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  CUSTOMER LIFETIME VALUE & REVENUE FORECASTING ENGINE")
    print("=" * 60)

    db = DatabaseManager(backend=args.db)
    try:
        result = run_pipeline(
            source=args.source,
            clv_horizon_months=args.clv_horizon,
            forecast_horizon_months=args.forecast_horizon,
            gross_margin=args.margin,
            db_manager=db,
            persist_to_db=True,
        )

        print("\n" + "-" * 60)
        print(f"✅ Pipeline Completed Successfully in {result.duration_seconds} seconds!")
        print(f"Run ID:            {result.run_id}")
        print(f"Dataset Source:    {result.metadata.source_name} (Synthetic: {result.metadata.is_synthetic})")
        print(f"Raw Transactions:  {result.dq_report.total_raw_rows:,}")
        print(f"Cleaned Sales:     {result.dq_report.final_cleaned_rows:,}")
        print(f"Unique Customers:  {len(result.customer_metrics_df):,}")
        print(f"Total Revenue:     ${result.customer_metrics_df['total_revenue'].sum():,.2f}")
        print(f"Repeat Rate:       {result.repeat_kpis.get('Repeat Purchase Rate (%)', 0.0):.1f}%")
        print(f"CLV Model:         {result.clv_diagnostics.model_name}")
        print(f"Report Generated:  {result.report_path}")
        print("-" * 60 + "\n")

    except Exception as e:
        logger.exception(f"Pipeline execution encountered an error: {e}")
        print(f"\n❌ Pipeline execution failed: {e}\n")
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    main()
