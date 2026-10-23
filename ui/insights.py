"""UI Page 7: Insights & Export Center.

Presents evidence-backed strategic recommendations, executive summaries,
and provides multi-format downloadable reports and analytical CSV datasets.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

def render_insights(pipeline_result):
    """Renders Insights & Export Center page."""
    st.markdown("## 💡 Strategic Insights & Export Center")
    st.caption("Data-driven business recommendations backed by empirical evidence, and downloadable analytical data extracts.")

    # 1. Strategic Recommendations Cards
    st.subheader("Data-Driven Strategic Recommendations")
    st.caption("Each recommendation is derived from deterministic rules evaluated against observed customer behavior and revenue distributions.")

    recommendations = pipeline_result.recommendations
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            r = rec.to_dict() if hasattr(rec, "to_dict") else rec
            priority = r.get("priority", "Medium")
            badge_color = "🔴" if priority == "High" else ("🟡" if priority == "Medium" else "🟢")

            with st.container():
                st.markdown(f"#### {badge_color} {idx}. {r.get('title')}")
                st.markdown(f"**Category:** `{r.get('category')}` | **Priority:** `{priority}`")

                c_rec1, c_rec2 = st.columns(2)
                with c_rec1:
                    st.info(f"**📊 Observed Evidence:**\n\n{r.get('observed_evidence')}")
                    st.caption(f"**Trigger Rule:** `{r.get('threshold_rule')}`")

                with c_rec2:
                    st.success(f"**🚀 Suggested Action:**\n\n{r.get('suggested_action')}")
                    st.warning(f"**⚠️ Strategic Limitation:**\n\n{r.get('limitation')}")

                st.markdown("---")
    else:
        st.info("No rule-based recommendations were triggered under current thresholds.")

    # 2. Executive Analytics Briefing
    st.subheader("Executive Analytics Summary")
    with st.expander("📄 View Full Generated Executive Markdown Report", expanded=False):
        st.markdown(pipeline_result.report_markdown)

    st.markdown("---")

    # 3. Downloadable Reports and Datasets
    st.subheader("Export Center: Download Analytical Datasets & Reports")
    st.caption("All datasets and reports are dynamically generated from calculated outputs.")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### 📑 Formal Reports")
        # Markdown Report Download
        st.download_button(
            label="📥 Download Executive Business Report (.md)",
            data=pipeline_result.report_markdown.encode("utf-8"),
            file_name=f"executive_business_report_{pipeline_result.run_id}.md",
            mime="text/markdown",
            use_container_width=True
        )

        # Data Quality Report CSV
        dq_csv = pipeline_result.dq_report.to_dataframe().to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Data Quality Health Audit (CSV)",
            data=dq_csv,
            file_name=f"data_quality_report_{pipeline_result.run_id}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_d2:
        st.markdown("#### 📊 Analytical Tables")
        # Customer-Level Metrics & CLV CSV
        cust_csv = pipeline_result.customer_metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Customer-Level Metrics & CLV (CSV)",
            data=cust_csv,
            file_name="customer_lifetime_value_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Revenue Forecast CSV
        fut_forecast = pipeline_result.forecast_results.future_forecasts
        if not fut_forecast.empty:
            fc_csv = fut_forecast.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Multi-Model Revenue Forecasts (CSV)",
                data=fc_csv,
                file_name="revenue_monthly_forecasts.csv",
                mime="text/csv",
                use_container_width=True
            )

        # RFM Segment Summary CSV
        seg_csv = pipeline_result.segment_summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download RFM Segment Performance (CSV)",
            data=seg_csv,
            file_name="rfm_segment_summary.csv",
            mime="text/csv",
            use_container_width=True
        )
