"""UI Page 2: Customer Segmentation.

Explores RFM scoring, behavioral segment distributions, Recency vs Frequency dynamics,
and granular customer-level profiles with export capabilities.
"""

import streamlit as st
import plotly.express as px
import pandas as pd

def render_segmentation(pipeline_result):
    """Renders Customer Segmentation page."""
    st.markdown("## 👥 Customer RFM Segmentation Studio")
    st.caption("Quantile-based Recency, Frequency, and Monetary scoring classifying customers into actionable behavioral cohorts.")

    cust_df = pipeline_result.customer_metrics_df
    seg_summary = pipeline_result.segment_summary_df

    if cust_df.empty or seg_summary.empty:
        st.warning("Segmentation data is unavailable. Please run the data pipeline.")
        return

    # 1. Segment Performance Summary Table
    st.subheader("Segment Performance & Contribution Summary")
    
    # Format table for clean display
    display_summary = seg_summary.copy()
    display_summary["Customer Count"] = display_summary["Customer Count"].apply(lambda x: f"{x:,}")
    display_summary["Customer Share (%)"] = display_summary["Customer Share (%)"].apply(lambda x: f"{x:.1f}%")
    display_summary["Total Revenue"] = display_summary["Total Revenue"].apply(lambda x: f"${x:,.2f}")
    display_summary["Revenue Share (%)"] = display_summary["Revenue Share (%)"].apply(lambda x: f"{x:.1f}%")
    display_summary["Avg AOV"] = display_summary["Avg AOV"].apply(lambda x: f"${x:,.2f}")
    display_summary["Avg Recency (Days)"] = display_summary["Avg Recency (Days)"].apply(lambda x: f"{x:.1f} d")

    st.dataframe(
        display_summary[["Segment", "Customer Count", "Customer Share (%)", "Total Revenue", "Revenue Share (%)", "Avg AOV", "Avg Recency (Days)", "Description"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # 2. Visualizations
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.subheader("Recency vs. Purchase Frequency")
        fig_scatter = px.scatter(
            cust_df,
            x="recency_days",
            y="num_invoices",
            color="segment",
            size="total_revenue",
            hover_name="CustomerID",
            hover_data={
                "recency_days": ":.1f",
                "num_invoices": ":d",
                "total_revenue": ":$.2f",
                "avg_order_value": ":$.2f",
                "segment": True
            },
            labels={
                "recency_days": "Days Since Last Purchase (Recency)",
                "num_invoices": "Distinct Invoices (Frequency)",
                "total_revenue": "Total Spend",
                "segment": "Segment"
            },
            title="Recency vs Frequency (Bubble size = Total Spend)"
        )
        fig_scatter.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_v2:
        st.subheader("Monetary Spend Distribution by Segment")
        fig_box = px.box(
            cust_df,
            x="segment",
            y="total_revenue",
            color="segment",
            log_y=True,
            labels={"segment": "Segment", "total_revenue": "Spend ($ Log Scale)"},
            title="Revenue Distribution per Segment (Log Scale)"
        )
        fig_box.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # 3. Interactive Filter & Customer Directory
    st.subheader("Searchable Customer Directory")
    
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        available_segments = sorted(list(cust_df["segment"].unique()))
        selected_segments = st.multiselect(
            "Filter by Customer Segment(s):",
            options=available_segments,
            default=available_segments
        )
    with col_f2:
        min_spend = st.number_input("Minimum Total Spend ($):", min_value=0.0, value=0.0, step=50.0)

    # Filtered DataFrame
    filtered_df = cust_df[
        (cust_df["segment"].isin(selected_segments)) &
        (cust_df["total_revenue"] >= min_spend)
    ].copy()

    st.caption(f"Showing **{len(filtered_df):,}** customers matching filters (out of {len(cust_df):,} total).")

    # Format for display table
    display_cust_df = filtered_df[[
        "CustomerID", "segment", "rfm_score", "recency_score", "frequency_score", "monetary_score",
        "num_invoices", "total_quantity", "total_revenue", "avg_order_value", "recency_days", "customer_lifespan_days"
    ]].copy()

    display_cust_df["total_revenue"] = display_cust_df["total_revenue"].apply(lambda x: f"${x:,.2f}")
    display_cust_df["avg_order_value"] = display_cust_df["avg_order_value"].apply(lambda x: f"${x:,.2f}")
    display_cust_df["recency_days"] = display_cust_df["recency_days"].apply(lambda x: f"{x:.1f}")
    display_cust_df["customer_lifespan_days"] = display_cust_df["customer_lifespan_days"].apply(lambda x: f"{x:.1f}")

    st.dataframe(display_cust_df, use_container_width=True, hide_index=True)

    # Export Button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Customers (CSV)",
        data=csv_bytes,
        file_name="rfm_segmented_customers.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # 4. Individual Customer Deep-Dive
    st.subheader("🔍 Individual Customer Behavioral Deep-Dive")
    customer_ids = sorted(list(filtered_df["CustomerID"].astype(str).unique()))
    if customer_ids:
        selected_cid = st.selectbox("Select Customer ID to inspect:", options=customer_ids)
        c_row = filtered_df[filtered_df["CustomerID"] == selected_cid].iloc[0]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Assigned Segment", c_row["segment"])
        d2.metric("Composite RFM Score", str(c_row["rfm_score"]))
        d3.metric("Total Historical Spend", f"${c_row['total_revenue']:,.2f}")
        d4.metric("Days Since Last Order", f"{c_row['recency_days']:.1f} days")

        d5, d6, d7, d8 = st.columns(4)
        d5.metric("Completed Invoices", f"{c_row['num_invoices']} orders")
        d6.metric("Average Order Value", f"${c_row['avg_order_value']:,.2f}")
        d7.metric("Observed Lifespan", f"{c_row['customer_lifespan_days']:.1f} days")
        interp = c_row.get("avg_interpurchase_time", 0.0)
        d8.metric("Avg Interpurchase Time", f"{interp:.1f} days" if interp > 0 else "N/A (Single Order)")

        # Customer transactions drilldown from cleaned sales
        sales_df = pipeline_result.cleaned_sales_df
        cust_tx = sales_df[sales_df["CustomerID"] == selected_cid]
        if not cust_tx.empty:
            st.markdown(f"**Transaction Item History for Customer `{selected_cid}` ({len(cust_tx)} line items):**")
            st.dataframe(
                cust_tx[["InvoiceNo", "InvoiceDate", "StockCode", "Description", "Quantity", "UnitPrice", "TotalAmount", "Country"]],
                use_container_width=True,
                hide_index=True
            )
