"""Customer Lifetime Value (CLV) & Revenue Forecasting Engine.

Main Streamlit Application Entrypoint.
Coordinates navigation across 7 analytical pages, maintains session state,
and applies custom professional CSS styling.
"""

from datetime import datetime
from pathlib import Path
import sys
import streamlit as st
import pandas as pd

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.orchestrator import run_pipeline, PipelineExecutionResult
from database.db_manager import DatabaseManager
from config.logging_config import get_logger

# Import UI Modules
from ui.overview import render_overview
from ui.segmentation import render_segmentation
from ui.clv_studio import render_clv_studio
from ui.forecasting import render_forecasting
from ui.retention import render_retention
from ui.data_quality import render_data_quality
from ui.insights import render_insights

logger = get_logger("app")

# Page Configuration
st.set_page_config(
    page_title="CLV & Revenue Forecasting Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Portfolio Grade CSS)
st.markdown("""
<style>
    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 14px 18px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #495057 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #212529 !important;
    }
    /* Sidebar Polish */
    section[data-testid="stSidebar"] {
        background-color: #fdfdfd;
        border-right: 1px solid #e0e0e0;
    }
    /* Buttons */
    button[kind="primary"] {
        background-color: #0d6efd;
        border-color: #0d6efd;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def load_or_init_pipeline() -> PipelineExecutionResult:
    """Loads pipeline results from session state or executes initial pipeline run."""
    if "pipeline_result" not in st.session_state:
        with st.spinner("Initializing analytical pipeline and loading data..."):
            db = DatabaseManager()
            result = run_pipeline(db_manager=db, persist_to_db=True)
            st.session_state["pipeline_result"] = result
            db.close()
    return st.session_state["pipeline_result"]

def on_pipeline_rerun(new_result: PipelineExecutionResult):
    """Callback to update session state when pipeline is rerun with new data."""
    st.session_state["pipeline_result"] = new_result

def main():
    # Load pipeline state
    pipeline_result = load_or_init_pipeline()

    # Sidebar Navigation
    st.sidebar.title("📈 CLV Engine")
    st.sidebar.caption("Portfolio Analytics Platform")

    pages = {
        "📊 Executive Overview": render_overview,
        "👥 Customer Segmentation": render_segmentation,
        "🔮 CLV Prediction Studio": render_clv_studio,
        "📈 Revenue Forecasting": render_forecasting,
        "🔄 Retention & Cohorts": render_retention,
        "🛠️ Data Quality & Pipeline": lambda res: render_data_quality(res, on_pipeline_rerun=on_pipeline_rerun),
        "💡 Insights & Export Center": render_insights,
    }

    selected_page_name = st.sidebar.radio(
        "Navigation",
        options=list(pages.keys()),
        index=0,
    )

    st.sidebar.markdown("---")

    # Sidebar Provenance Card
    meta = pipeline_result.metadata
    st.sidebar.markdown("### 📌 Active Source")
    if meta.is_synthetic:
        st.sidebar.warning("⚠️ **Synthetic Dataset Active**\n\n(Deterministic simulation with anomalies)")
    else:
        st.sidebar.success(f"✅ **{meta.source_name}**")

    st.sidebar.caption(f"**Run ID:** `{pipeline_result.run_id}`")
    st.sidebar.caption(f"**Customers:** {len(pipeline_result.customer_metrics_df):,} | **Sales Rows:** {pipeline_result.dq_report.final_cleaned_rows:,}")
    st.sidebar.caption(f"**Database:** `{DatabaseManager().backend.upper()}`")

    # Render chosen page
    page_func = pages[selected_page_name]
    page_func(pipeline_result)

if __name__ == "__main__":
    main()
