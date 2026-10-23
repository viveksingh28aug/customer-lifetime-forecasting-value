"""UI Page 3: CLV Prediction Studio.

Interactive modeling workspace for estimating Customer Lifetime Value via Probabilistic (BG/NBD + Gamma-Gamma)
and Transparent Baseline models with configurable horizons, margins, and diagnostics.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.clv.baseline import compute_historical_baseline_clv, compute_transparent_forecast_clv
from src.clv.probabilistic import fit_probabilistic_clv
from src.clv.evaluation import summarize_clv_distribution, compare_clv_models

def render_clv_studio(pipeline_result):
    """Renders CLV Prediction Studio page."""
    st.markdown("## 🔮 Customer Lifetime Value (CLV) Prediction Studio")
    st.caption("Predict future customer transactions and monetary value over configurable horizons with model diagnostics and transparent assumptions.")

    cust_df = pipeline_result.customer_metrics_df
    sales_df = pipeline_result.cleaned_sales_df

    if cust_df.empty:
        st.warning("Customer data is not available. Please run the pipeline first.")
        return

    # 1. Model Controls & Configuration Bar
    st.markdown("### Model Configuration & Horizon Parameters")
    c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns(4)

    with c_ctrl1:
        model_choice = st.selectbox(
            "Select CLV Methodology:",
            options=[
                "Probabilistic (BG/NBD + Gamma-Gamma)",
                "Transparent Forecast Baseline",
                "Historical Baseline"
            ],
            index=0
        )

    with c_ctrl2:
        horizon_choice = st.selectbox(
            "Forecast Horizon (Months):",
            options=[3, 6, 12],
            index=2
        )

    with c_ctrl3:
        gross_margin = st.slider(
            "Gross Profit Margin (%):",
            min_value=10,
            max_value=100,
            value=100,
            step=5,
            help="100% represents Gross Revenue CLV; lower margins represent Net Profit CLV."
        ) / 100.0

    with c_ctrl4:
        discount_rate = st.slider(
            "Annual Discount Rate (%):",
            min_value=0,
            max_value=25,
            value=10,
            step=1,
            help="Discount rate for present-value discounting of future cash flows."
        ) / 100.0

    # Compute or select model predictions based on UI choices
    with st.spinner("Calculating CLV estimates..."):
        if model_choice == "Probabilistic (BG/NBD + Gamma-Gamma)":
            clv_df, diag = fit_probabilistic_clv(
                df_sales=sales_df,
                customer_metrics_df=cust_df,
                horizon_months=horizon_choice,
                discount_rate=discount_rate,
                gross_margin=gross_margin,
                run_id=pipeline_result.run_id
            )
        elif model_choice == "Transparent Forecast Baseline":
            clv_df = compute_transparent_forecast_clv(
                customer_metrics_df=cust_df,
                horizon_months=horizon_choice,
                discount_rate=discount_rate,
                gross_margin=gross_margin,
                recency_decay_factor=True,
                run_id=pipeline_result.run_id
            )
            from src.clv.probabilistic import CLVModelDiagnostics
            diag = CLVModelDiagnostics(
                model_name="Transparent Forecast Baseline",
                is_probabilistic=False,
                assumptions_met=True,
                repeat_customers_count=int((cust_df["num_invoices"] > 1).sum()),
                convergence_status="Direct Analytical Computation",
                limitations=["Extrapolates observed order frequency with exponential recency damping."],
            )
        else:
            clv_df = compute_historical_baseline_clv(
                customer_metrics_df=cust_df,
                gross_margin=gross_margin,
                horizon_months=horizon_choice,
                run_id=pipeline_result.run_id
            )
            from src.clv.probabilistic import CLVModelDiagnostics
            diag = CLVModelDiagnostics(
                model_name="Historical Baseline",
                is_probabilistic=False,
                assumptions_met=True,
                repeat_customers_count=int((cust_df["num_invoices"] > 1).sum()),
                convergence_status="Direct Historical Annualization",
                limitations=["Assumes past annual spend will repeat without accounting for churn risk."],
            )

    # 2. Model Diagnostics & Status Banner
    st.markdown("---")
    if diag.is_probabilistic:
        st.success(
            f"✅ **ACTIVE MODEL: {diag.model_name}** | Status: `{diag.convergence_status}` | "
            f"Repeat Buyers: {diag.repeat_customers_count} | Freq-Monetary Corr: r={diag.correlation_freq_monetary} (p={diag.correlation_p_value})",
            icon="🎯"
        )
    else:
        st.info(
            f"ℹ️ **ACTIVE MODEL: {diag.model_name}** | Status: `{diag.convergence_status}`"
            + (f" | *Fallback Reason:* {diag.fallback_reason}" if diag.fallback_reason else ""),
            icon="ℹ️"
        )

    # 3. High-Level CLV Summary Metrics
    clv_stats = summarize_clv_distribution(clv_df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"Total Predicted Value ({horizon_choice}M)", f"${clv_stats.get('Total Predicted CLV', 0.0):,.2f}")
    m2.metric("Mean Customer CLV", f"${clv_stats.get('Mean CLV', 0.0):,.2f}")
    m3.metric("Median Customer CLV", f"${clv_stats.get('Median CLV', 0.0):,.2f}")
    m4.metric("Top 10% Customer Share", f"{clv_stats.get('Top 10% Customer Value Share (%)', 0.0):.1f}%")
    m5.metric("Max Single Customer CLV", f"${clv_stats.get('Max CLV', 0.0):,.2f}")

    # 4. Charts: Distribution & Historical vs Predicted
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("Predicted Customer Lifetime Value Distribution")
        fig_hist = px.histogram(
            clv_df,
            x="predicted_clv",
            nbins=35,
            labels={"predicted_clv": "Predicted CLV ($)"},
            title=f"Distribution of Predicted {horizon_choice}-Month Customer Value",
            color_discrete_sequence=["#2ca02c"]
        )
        fig_hist.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Customer Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_c2:
        st.subheader("Historical Spend vs. Predicted Future Value")
        # Ensure total_historical_revenue is present
        if "total_historical_revenue" not in clv_df.columns and "total_revenue" in cust_df.columns:
            rev_map = dict(zip(cust_df["CustomerID"], cust_df["total_revenue"]))
            clv_df["total_historical_revenue"] = clv_df["CustomerID"].map(rev_map)

        fig_scatter = px.scatter(
            clv_df,
            x="total_historical_revenue",
            y="predicted_clv",
            hover_name="CustomerID",
            color="segment" if "segment" in clv_df.columns else None,
            labels={"total_historical_revenue": "Historical Total Spend ($)", "predicted_clv": f"Predicted {horizon_choice}M CLV ($)"},
            title="Historical Spend vs Predicted Future Value"
        )
        fig_scatter.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 5. Top 10 Customers by Projected Future Value
    st.subheader(f"Top 10 Customers by Projected {horizon_choice}-Month CLV")
    top_10_clv = clv_df.sort_values(by="predicted_clv", ascending=False).head(10).copy()

    fig_top = px.bar(
        top_10_clv,
        x="CustomerID",
        y="predicted_clv",
        text="predicted_clv",
        color="predicted_clv",
        color_continuous_scale="Teal",
        labels={"CustomerID": "Customer ID", "predicted_clv": "Predicted CLV ($)"},
        title=f"Top 10 Highest Value Customer Forecasts ({horizon_choice} Months)"
    )
    fig_top.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
    fig_top.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig_top, use_container_width=True)

    # 6. Searchable Predictions Table & CSV Download
    st.subheader("Granular Customer Lifetime Value Forecast Table")
    display_clv = clv_df.sort_values(by="predicted_clv", ascending=False).copy()
    
    # Format currency columns
    fmt_df = display_clv.copy()
    fmt_df["predicted_clv"] = fmt_df["predicted_clv"].apply(lambda x: f"${x:,.2f}")
    if "predicted_revenue" in fmt_df.columns:
        fmt_df["predicted_revenue"] = fmt_df["predicted_revenue"].apply(lambda x: f"${x:,.2f}")
    if "predicted_spend_per_tx" in fmt_df.columns:
        fmt_df["predicted_spend_per_tx"] = fmt_df["predicted_spend_per_tx"].apply(lambda x: f"${x:,.2f}")
    if "total_historical_revenue" in fmt_df.columns:
        fmt_df["total_historical_revenue"] = fmt_df["total_historical_revenue"].apply(lambda x: f"${x:,.2f}")

    st.dataframe(fmt_df, use_container_width=True, hide_index=True)

    # Download Button
    csv_bytes = clv_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download {model_choice} Predictions (CSV)",
        data=csv_bytes,
        file_name=f"clv_predictions_{horizon_choice}m.csv",
        mime="text/csv",
    )

    # 7. Model Card & Limitations Expander
    with st.expander("ℹ️ Model Card, Theoretical Assumptions & Limitations"):
        st.markdown(f"""
        ### Methodology: {diag.model_name}
        - **Horizon:** {horizon_choice} Months
        - **Profit Margin Applied:** {gross_margin * 100:.0f}% ({'Gross Revenue Basis' if gross_margin == 1.0 else 'Net Profit Margin'})
        - **Annual Discount Rate:** {discount_rate * 100:.1f}%

        #### Assumptions & Failure Modes:
        """)
        for lim in diag.limitations:
            st.markdown(f"- {lim}")

        st.markdown("""
        #### Critical Disclosure:
        Customer Lifetime Value estimates are statistical projections based on past purchasing behavior.
        They do not represent guaranteed cash flows. Business macroeconomic shifts, pricing changes,
        or competitive attrition may cause realized performance to diverge from predictions.
        """)
