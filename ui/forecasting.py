"""UI Page 4: Revenue Forecasting.

Presents aggregate monthly sales time-series modeling, chronological holdout evaluation,
multi-model comparative metrics (MAE, RMSE, WAPE), and future projections with prediction intervals.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.forecasting.revenue_forecast import run_revenue_forecast_pipeline

def render_forecasting(pipeline_result):
    """Renders Revenue Forecasting page."""
    st.markdown("## 📈 Monthly Revenue Forecasting Engine")
    st.caption("Time-series projection and chronological holdout validation comparing Naive, Seasonal-Naive, and Holt-Winters models.")

    sales_df = pipeline_result.cleaned_sales_df
    forecast_res = pipeline_result.forecast_results

    if forecast_res.monthly_history.empty:
        st.warning("Insufficient monthly transaction data to perform forecasting.")
        return

    # 1. Forecast Controls
    st.markdown("### Horizon & Validation Settings")
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        selected_horizon = st.selectbox(
            "Forecast Horizon (Months ahead):",
            options=[3, 6, 12],
            index=1 if forecast_res.horizon_months == 6 else (0 if forecast_res.horizon_months == 3 else 2)
        )
    with col_ctrl2:
        selected_model = st.selectbox(
            "Primary Projection Model:",
            options=["Holt-Winters", "Seasonal-Naive", "Naive Baseline"],
            index=0
        )

    # Re-run forecasting pipeline if user changes horizon from default
    if selected_horizon != forecast_res.horizon_months:
        with st.spinner("Re-computing time-series forecast..."):
            forecast_res = run_revenue_forecast_pipeline(
                df_sales=sales_df,
                horizon_months=selected_horizon,
                run_id=pipeline_result.run_id
            )

    # 2. Chronological Validation Performance Metrics
    st.markdown("---")
    st.subheader(f"Chronological Holdout Validation ({forecast_res.test_months} Months Held-Out)")
    st.caption("Models were fitted strictly on past training months and evaluated against unseen test actuals.")

    metrics_df = forecast_res.metrics_summary
    if not metrics_df.empty:
        # Metrics summary table
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Highlight primary model metrics
        model_sub = metrics_df[metrics_df["Model"].str.contains(selected_model, case=False, na=False)]
        if not model_sub.empty:
            m_row = model_sub.iloc[0]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Mean Absolute Error (MAE)", f"${m_row['MAE']:,.2f}" if pd.notna(m_row.get("MAE")) else "N/A")
            k2.metric("Root Mean Squared Error (RMSE)", f"${m_row['RMSE']:,.2f}" if pd.notna(m_row.get("RMSE")) else "N/A")
            k3.metric("Weighted Absolute % Error (WAPE)", f"{m_row['WAPE (%)']:.1f}%" if pd.notna(m_row.get("WAPE (%)")) else "N/A")
            k4.metric("Mean Absolute % Error (MAPE)", f"{m_row['MAPE (%)']:.1f}%" if pd.notna(m_row.get("MAPE (%)")) else "N/A")

    # 3. Main Chart: Historical Actuals + Chronological Test + Future Projections
    st.markdown("---")
    st.subheader("Historical Sales, Validation Test Period, and Future Projection")

    hist_df = forecast_res.monthly_history.copy()
    test_eval_df = forecast_res.test_evaluations.copy()
    fut_df = forecast_res.future_forecasts[forecast_res.future_forecasts["ModelName"] == selected_model].copy()

    fig = go.Figure()

    # Split train and test actuals
    train_n = forecast_res.train_months
    train_actuals = hist_df.iloc[:train_n]
    test_actuals = hist_df.iloc[train_n:]

    # Trace 1: Historical Training Actuals
    fig.add_trace(go.Scatter(
        x=train_actuals["PeriodDate"],
        y=train_actuals["TotalRevenue"],
        mode="lines+markers",
        name="Training Actuals",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=6)
    ))

    # Trace 2: Held-out Test Actuals
    if not test_actuals.empty:
        fig.add_trace(go.Scatter(
            x=test_actuals["PeriodDate"],
            y=test_actuals["TotalRevenue"],
            mode="lines+markers",
            name="Test Actuals (Held Out)",
            line=dict(color="#2ca02c", width=3, dash="dot"),
            marker=dict(size=8, symbol="diamond")
        ))

    # Trace 3: Model Prediction on Test Period
    if not test_eval_df.empty:
        pred_col = "Holt_Winters_Prediction" if selected_model == "Holt-Winters" else (
            "Seasonal_Naive_Prediction" if selected_model == "Seasonal-Naive" else "Naive_Prediction"
        )
        if pred_col in test_eval_df.columns:
            fig.add_trace(go.Scatter(
                x=test_eval_df["PeriodDate"],
                y=test_eval_df[pred_col],
                mode="lines+markers",
                name=f"{selected_model} (Test Fit)",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                marker=dict(size=6, symbol="x")
            ))

    # Trace 4: Future Forecast
    if not fut_df.empty:
        # Connect last actual to first forecast for visual continuity
        fut_dates = list(fut_df["PeriodDate"])
        fut_vals = list(fut_df["PredictedRevenue"])
        fut_lower = list(fut_df["LowerBound"])
        fut_upper = list(fut_df["UpperBound"])

        fig.add_trace(go.Scatter(
            x=fut_dates,
            y=fut_vals,
            mode="lines+markers",
            name=f"Future Forecast ({selected_model})",
            line=dict(color="#9467bd", width=3),
            marker=dict(size=8)
        ))

        # Prediction interval shading
        fig.add_trace(go.Scatter(
            x=fut_dates + fut_dates[::-1],
            y=fut_upper + fut_lower[::-1],
            fill="toself",
            fillcolor="rgba(148, 103, 189, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Prediction Interval"
        ))

    fig.update_layout(
        title=f"Monthly Revenue Trajectory & {selected_horizon}-Month Forecast ({selected_model})",
        xaxis_title="Date",
        yaxis_title="Monthly Revenue ($)",
        height=480,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. Future Projections Table & Download
    st.subheader(f"Future {selected_horizon}-Month Projections Data Table")
    fut_display = fut_df[["YearMonth", "ModelName", "PredictedRevenue", "LowerBound", "UpperBound"]].copy()
    fut_display["PredictedRevenue"] = fut_display["PredictedRevenue"].apply(lambda x: f"${x:,.2f}")
    fut_display["LowerBound"] = fut_display["LowerBound"].apply(lambda x: f"${x:,.2f}")
    fut_display["UpperBound"] = fut_display["UpperBound"].apply(lambda x: f"${x:,.2f}")

    st.dataframe(fut_display, use_container_width=True, hide_index=True)

    # CSV Download
    csv_bytes = fut_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download {selected_model} Forecast (CSV)",
        data=csv_bytes,
        file_name=f"revenue_forecast_{selected_horizon}m.csv",
        mime="text/csv",
    )

    # 5. Methodological Notes
    st.markdown("---")
    st.markdown("#### Methodological Notes & Chronological Split Details")
    for note in forecast_res.notes:
        st.markdown(f"- {note}")
