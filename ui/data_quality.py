"""UI Page 6: Data Quality & Pipeline Orchestration.

Manages dataset ingestion, manual uploads (CSV/XLSX), UCI public dataset download,
synthetic dataset regeneration, schema mapping, data quality auditing, and pipeline reruns.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from src.ingestion.loader import get_uci_download_instructions, download_uci_dataset
from src.ingestion.schema_validator import validate_and_map_schema
from src.pipeline.orchestrator import run_pipeline
from scripts.generate_synthetic_data import generate_synthetic_retail_data
from config.settings import REQUIRED_COLUMNS, RAW_DATA_DIR, SYNTHETIC_DATA_DIR
from database.db_manager import DatabaseManager

def render_data_quality(pipeline_result, on_pipeline_rerun=None):
    """Renders Data Quality & Pipeline Management page."""
    st.markdown("## 🛠️ Data Quality & Pipeline Ingestion Center")
    st.caption("Manage transaction data sources, inspect data cleaning audit metrics, and execute the analytical pipeline.")

    # 1. Pipeline Execution & Ingestion Status
    dq = pipeline_result.dq_report
    meta = pipeline_result.metadata

    st.subheader("Data Quality Health Audit Report")
    
    # Top KPI cards
    col_dq1, col_dq2, col_dq3, col_dq4 = st.columns(4)
    col_dq1.metric("Data Quality Health Score", f"{dq.data_quality_score:.1f}%")
    col_dq2.metric("Total Raw Ingested Rows", f"{dq.total_raw_rows:,}")
    col_dq3.metric("Final Valid Sales Rows", f"{dq.final_cleaned_rows:,}")
    col_dq4.metric("Excluded Anomaly Records", f"{dq.excluded_rows:,}")

    # Detailed Audit Breakdown Table
    st.markdown("#### Quality Audit Breakdown")
    dq_df = dq.to_dataframe()
    st.dataframe(dq_df, use_container_width=True, hide_index=True)

    st.caption(f"**Applied Cleaning Policy:** `{dq.cleaning_policy}`")

    st.markdown("---")

    # 2. Dataset Management & Ingestion Controls
    st.subheader("Dataset Ingestion & Upload Hub")
    st.caption("Switch between synthetic benchmarks, download the official UCI dataset, or upload a custom retail transaction file.")

    tab_upload, tab_uci, tab_synthetic = st.tabs([
        "📤 Upload Custom Dataset (CSV / XLSX)",
        "🌐 UCI Online Retail (Public Dataset)",
        "🎲 Synthetic Retail Generator"
    ])

    # Tab 1: Manual File Upload
    with tab_upload:
        st.markdown("#### Upload Transaction File")
        uploaded_file = st.file_uploader(
            "Upload Retail Transactions (.csv, .xlsx, .xls)",
            type=["csv", "xlsx", "xls"],
            help="File must contain transactional fields such as InvoiceNo, Quantity, UnitPrice, InvoiceDate, CustomerID."
        )

        if uploaded_file is not None:
            # Preview raw data
            try:
                if uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
                    df_preview = pd.read_excel(uploaded_file, nrows=10)
                else:
                    df_preview = pd.read_csv(uploaded_file, nrows=10)

                st.markdown(f"**Preview of `{uploaded_file.name}` (First 10 rows):**")
                st.dataframe(df_preview, use_container_width=True)

                # Column mapping helper
                st.markdown("##### Column Mapping Verification")
                schema_test = validate_and_map_schema(df_preview)
                if schema_test.is_valid:
                    st.success("✅ Standard columns detected and mapped automatically!")
                else:
                    st.warning(f"⚠️ Unmapped mandatory columns detected: {schema_test.missing_columns}")
                    st.caption("Please ensure your dataset columns correspond to standard retail transaction headers.")

                if st.button("🚀 Process & Run Pipeline on Uploaded Dataset", type="primary"):
                    progress_bar = st.progress(0, text="Initializing pipeline...")

                    def update_ui_progress(pct, msg):
                        progress_bar.progress(int(pct * 100), text=msg)

                    new_res = run_pipeline(
                        source=uploaded_file,
                        file_type="excel" if (uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls")) else "csv",
                        is_synthetic=False,
                        source_name=f"Uploaded: {uploaded_file.name}",
                        progress_callback=update_ui_progress
                    )
                    st.success("🎉 Pipeline executed successfully on uploaded data!")
                    if on_pipeline_rerun:
                        on_pipeline_rerun(new_res)
                    st.rerun()

            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")

    # Tab 2: UCI Dataset Management
    with tab_uci:
        st.markdown(get_uci_download_instructions())

        c_uci1, c_uci2 = st.columns(2)
        with c_uci1:
            if st.button("🌐 Attempt Automated Download from UCI Archive"):
                with st.spinner("Downloading from UCI Machine Learning Repository..."):
                    downloaded_file = download_uci_dataset(timeout=25)
                    if downloaded_file and downloaded_file.exists():
                        st.success(f"Downloaded UCI dataset to: `{downloaded_file}`")
                        if st.button("Run Pipeline on Downloaded UCI Dataset"):
                            new_res = run_pipeline(source=downloaded_file, is_synthetic=False, source_name="UCI Online Retail")
                            if on_pipeline_rerun:
                                on_pipeline_rerun(new_res)
                            st.rerun()
                    else:
                        st.warning("Automated direct download timed out or network blocked. Please manually download the file using the instructions above and upload it in the Upload tab.")

    # Tab 3: Synthetic Data Generator
    with tab_synthetic:
        st.markdown("#### Deterministic Synthetic Data Benchmark")
        st.caption("Generates simulated retail transactions with calibrated customer activity and intentional data quality anomalies.")

        sc1, sc2, sc3 = st.columns(3)
        num_custs = sc1.number_input("Synthetic Customers:", min_value=50, max_value=2000, value=400, step=50)
        num_invs = sc2.number_input("Synthetic Orders / Invoices:", min_value=500, max_value=10000, value=2500, step=250)
        seed_val = sc3.number_input("Deterministic Random Seed:", min_value=1, max_value=9999, value=42, step=1)

        if st.button("🔄 Regenerate & Process Synthetic Dataset"):
            progress_bar = st.progress(0, text="Generating deterministic synthetic dataset...")
            generate_synthetic_retail_data(
                num_customers=num_custs,
                num_invoices=num_invs,
                seed=seed_val
            )
            progress_bar.progress(30, text="Executing full pipeline on new synthetic data...")

            def update_ui_progress(pct, msg):
                progress_bar.progress(int(pct * 100), text=msg)

            new_res = run_pipeline(
                is_synthetic=True,
                source_name=f"Synthetic Retail (Seed {seed_val})",
                progress_callback=update_ui_progress
            )
            st.success("🎉 Synthetic dataset generated and processed successfully!")
            if on_pipeline_rerun:
                on_pipeline_rerun(new_res)
            st.rerun()

    st.markdown("---")

    # 3. Database Architecture & Table Inspection
    st.subheader("Analytical Database Backend Status")
    db = DatabaseManager()
    st.caption(f"Connected backend: `{db.backend.upper()}` | Storage Path: `{db.db_path or 'Default Path'}`")

    try:
        tables = ["pipeline_metadata", "data_quality_reports", "customer_metrics", "rfm_segments", "clv_predictions", "revenue_forecasts"]
        table_stats = []
        for tbl in tables:
            try:
                cnt_df = db.query(f"SELECT COUNT(*) as row_count FROM {tbl}")
                row_cnt = cnt_df["row_count"].iloc[0] if not cnt_df.empty else 0
                table_stats.append({"Table Name": tbl, "Stored Rows": f"{row_cnt:,}", "Status": "Active"})
            except Exception:
                table_stats.append({"Table Name": tbl, "Stored Rows": "0", "Status": "Empty / Initializing"})

        st.dataframe(pd.DataFrame(table_stats), use_container_width=True, hide_index=True)
    finally:
        db.close()
