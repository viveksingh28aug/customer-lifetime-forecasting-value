"""UI Page 5: Retention & Cohort Analysis.

Visualizes customer repeat purchasing behavior, monthly cohort retention matrices,
inactivity lifecycle distributions, and flags high-value at-risk accounts.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_retention(pipeline_result):
    """Renders Retention & Cohort Analysis page."""
    st.markdown("## 🔄 Customer Retention & Cohort Dynamics")
    st.caption("Empirical repeat purchasing metrics, acquisition cohort decay matrices, and operational inactivity indicators.")

    repeat_kpis = pipeline_result.repeat_kpis
    cohort_ret = pipeline_result.cohort_retention_df
    inactivity_df = pipeline_result.inactivity_dist_df
    at_risk_df = pipeline_result.at_risk_df
    cust_df = pipeline_result.customer_metrics_df

    # 1. Repeat Purchasing Headline KPIs
    st.subheader("Repeat Purchasing Performance")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Repeat Purchase Rate", f"{repeat_kpis.get('Repeat Purchase Rate (%)', 0.0):.1f}%")
    r2.metric("Repeat Customer Revenue Share", f"{repeat_kpis.get('Repeat Customers Revenue Share (%)', 0.0):.1f}%")
    r3.metric("Repeat Customers", f"{repeat_kpis.get('Repeat Customers', 0):,} accounts")
    r4.metric("Single-Purchase Customers", f"{repeat_kpis.get('Single-Purchase Customers', 0):,} accounts")

    c_sub1, c_sub2 = st.columns(2)
    c_sub1.caption(f"**Average Orders per Repeat Buyer:** {repeat_kpis.get('Avg Orders per Repeat Customer', 1.0):.1f} invoices")
    c_sub2.caption(f"**Average Lifetime Spend (Repeat vs Single):** ${repeat_kpis.get('Avg Spend Repeat Customer', 0.0):,.2f} vs ${repeat_kpis.get('Avg Spend Single Customer', 0.0):,.2f}")

    st.markdown("---")

    # 2. Monthly Cohort Retention Heatmap
    st.subheader("Monthly Acquisition Cohort Retention Matrix (%)")
    st.caption("Tracks the percentage of customers in each acquisition cohort who placed an order in subsequent months.")

    if not cohort_ret.empty:
        # Plotly Heatmap
        z_vals = cohort_ret.values
        x_labels = [f"Month {c}" for c in cohort_ret.columns]
        y_labels = list(cohort_ret.index)

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_vals,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            text=[[f"{val:.1f}%" if pd.notna(val) else "" for val in row] for row in z_vals],
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar=dict(title="Retention %")
        ))

        fig_heat.update_layout(
            height=max(360, len(y_labels) * 26),
            xaxis_title="Months Since Acquisition (Cohort Index)",
            yaxis_title="Acquisition Cohort (First Purchase)",
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Insufficient longitudinal data to construct cohort matrix.")

    st.markdown("---")

    # 3. Customer Inactivity Lifecycle Distribution
    st.subheader("Customer Inactivity Lifecycle Distribution")
    st.caption("Distribution of accounts based on elapsed days since their most recent transaction.")

    col_inact1, col_inact2 = st.columns([3, 2])

    with col_inact1:
        if not inactivity_df.empty:
            fig_inact = px.bar(
                inactivity_df,
                x="Inactivity_Bucket",
                y="Customer_Count",
                text="Customer_Share_Pct",
                color="Customer_Count",
                color_continuous_scale="Reds",
                labels={"Inactivity_Bucket": "Lifecycle Stage", "Customer_Count": "Customer Count"},
                title="Customer Counts across Recency Inactivity Stages"
            )
            fig_inact.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_inact.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_inact, use_container_width=True)

    with col_inact2:
        if not inactivity_df.empty:
            st.markdown("#### Inactivity Stage Revenue Breakdown")
            inact_disp = inactivity_df.copy()
            inact_disp["Total_Revenue"] = inact_disp["Total_Revenue"].apply(lambda x: f"${x:,.2f}")
            inact_disp["Customer_Share_Pct"] = inact_disp["Customer_Share_Pct"].apply(lambda x: f"{x:.1f}%")
            inact_disp["Revenue_Share_Pct"] = inact_disp["Revenue_Share_Pct"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(
                inact_disp[["Inactivity_Bucket", "Customer_Count", "Customer_Share_Pct", "Total_Revenue", "Revenue_Share_Pct"]],
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---")

    # 4. At-Risk High-Value Accounts
    st.subheader("⚠️ High-Value At-Risk Accounts (Heuristic Inactivity Alert)")
    st.caption(
        "Accounts with above-average historical spend that have remained dormant for > 90 days. "
        "Explicitly classified using a heuristic recency threshold rather than a black-box churn model."
    )

    if not at_risk_df.empty:
        disp_at_risk = at_risk_df[[
            "CustomerID", "segment", "num_invoices", "total_revenue", "recency_days", "Suggested_Action"
        ]].copy()
        disp_at_risk["total_revenue"] = disp_at_risk["total_revenue"].apply(lambda x: f"${x:,.2f}")
        disp_at_risk["recency_days"] = disp_at_risk["recency_days"].apply(lambda x: f"{x:.1f} d")

        st.dataframe(disp_at_risk, use_container_width=True, hide_index=True)

        csv_bytes = at_risk_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download At-Risk Customer Campaign List (CSV)",
            data=csv_bytes,
            file_name="at_risk_high_value_customers.csv",
            mime="text/csv",
        )
    else:
        st.success("No high-value customers currently exceed the 90-day dormancy threshold.")
