"""UI Page 1: Executive Overview.

Presents high-level business KPIs, monthly sales velocity trends,
customer segment revenue distribution, and provenance status.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def render_overview(pipeline_result):
    """Renders Executive Overview page."""
    st.markdown("## 📊 Executive Overview")
    st.caption("Strategic summary of customer lifetime value, aggregate sales trajectory, and segment contributions.")

    # 1. Active Dataset Provenance & Alerts
    meta = pipeline_result.metadata
    if meta.is_synthetic:
        st.warning(
            f"⚠️ **ACTIVE DATA SOURCE: SYNTHETIC DATASET** (`{meta.source_name}`). "
            "Data was generated deterministically with simulated customer behavior and anomalies. "
            "Metrics reflect synthetic modeling patterns and not actual enterprise figures.",
            icon="⚠️"
        )
    else:
        st.success(f"✅ **ACTIVE DATA SOURCE:** `{meta.source_name}` (Authentic Dataset)", icon="✅")

    # 2. Executive Metric Cards
    cust_df = pipeline_result.customer_metrics_df
    sales_df = pipeline_result.cleaned_sales_df
    forecast_res = pipeline_result.forecast_results
    repeat_kpis = pipeline_result.repeat_kpis
    clv_preds = pipeline_result.clv_predictions_df

    total_revenue = float(cust_df["total_revenue"].sum())
    unique_custs = len(cust_df)
    total_orders = int(sales_df["InvoiceNo"].nunique())
    aov = total_revenue / total_orders if total_orders > 0 else 0.0
    repeat_rate = repeat_kpis.get("Repeat Purchase Rate (%)", 0.0)
    avg_cust_rev = total_revenue / unique_custs if unique_custs > 0 else 0.0

    # CLV metric
    total_pred_clv = float(clv_preds["predicted_clv"].sum()) if not clv_preds.empty and "predicted_clv" in clv_preds.columns else 0.0

    # Latest Month and Forecasted Month Revenue
    latest_monthly_rev = 0.0
    if not forecast_res.monthly_history.empty:
        latest_monthly_rev = float(forecast_res.monthly_history["TotalRevenue"].iloc[-1])

    fut_forecast = forecast_res.future_forecasts
    next_month_rev = 0.0
    if not fut_forecast.empty:
        hw_rows = fut_forecast[fut_forecast["ModelName"] == "Holt-Winters"]
        if not hw_rows.empty:
            next_month_rev = float(hw_rows["PredictedRevenue"].iloc[0])
        else:
            next_month_rev = float(fut_forecast["PredictedRevenue"].iloc[0])

    # Row 1 KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Historical Revenue", f"${total_revenue:,.2f}")
    col2.metric("Unique Customers", f"{unique_custs:,}")
    col3.metric("Total Invoices / Orders", f"{total_orders:,}")
    col4.metric("Average Order Value (AOV)", f"${aov:,.2f}")

    # Row 2 KPIs
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Repeat Customer Rate", f"{repeat_rate:.1f}%")
    col6.metric("Avg Customer Lifetime Spend", f"${avg_cust_rev:,.2f}")
    col7.metric("Pipeline Estimated CLV", f"${total_pred_clv:,.2f}")
    delta_str = f"{((next_month_rev - latest_monthly_rev) / latest_monthly_rev * 100):+.1f}%" if latest_monthly_rev > 0 else None
    col8.metric("Next Month Forecast", f"${next_month_rev:,.2f}", delta=delta_str)

    st.markdown("---")

    # 3. Visualizations
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.subheader("Monthly Historical Revenue Velocity")
        if not forecast_res.monthly_history.empty:
            hist_df = forecast_res.monthly_history.copy()
            hist_df["YearMonth_Str"] = hist_df["YearMonth"].astype(str)
            fig_hist = px.line(
                hist_df,
                x="YearMonth_Str",
                y="TotalRevenue",
                markers=True,
                labels={"YearMonth_Str": "Month", "TotalRevenue": "Revenue ($)"},
                title="Historical Monthly Revenue Trend"
            )
            fig_hist.update_traces(line_color="#1f77b4", line_width=3, marker_size=7)
            fig_hist.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Insufficient monthly data to render revenue velocity.")

    with col_chart2:
        st.subheader("RFM Customer Segment Distribution")
        seg_summary = pipeline_result.segment_summary_df
        if not seg_summary.empty:
            fig_pie = px.pie(
                seg_summary,
                names="Segment",
                values="Customer Count",
                hole=0.45,
                title="Customer Share by Segment",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No segmentation distribution data available.")

    # Row 2 Charts
    st.subheader("Revenue Contribution by Customer Segment")
    if not seg_summary.empty:
        fig_bar = px.bar(
            seg_summary.sort_values(by="Total Revenue", ascending=True),
            x="Total Revenue",
            y="Segment",
            orientation="h",
            text="Revenue Share (%)",
            color="Total Revenue",
            color_continuous_scale="Viridis",
            labels={"Total Revenue": "Total Revenue ($)", "Segment": "Segment"},
            title="Segment Revenue Contribution and Percentage Share"
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Provenance metadata footer
    st.markdown("---")
    c_foot1, c_foot2, c_foot3 = st.columns(3)
    c_foot1.caption(f"**Pipeline Run ID:** `{pipeline_result.run_id}`")
    c_foot2.caption(f"**Data Cleaned Rows:** {len(sales_df):,} / {pipeline_result.dq_report.total_raw_rows:,}")
    c_foot3.caption(f"**Execution Runtime:** {pipeline_result.duration_seconds:.2f} seconds")
