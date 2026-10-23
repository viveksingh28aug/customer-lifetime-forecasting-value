-- Reusable Analytical SQL Queries for DuckDB / SQLite / PostgreSQL

-- 1. Executive Revenue & Customer KPI Summary
SELECT
    COUNT(DISTINCT customer_id) AS total_customers,
    COUNT(DISTINCT invoice_no) AS total_orders,
    ROUND(SUM(total_amount), 2) AS total_revenue,
    ROUND(SUM(total_amount) / COUNT(DISTINCT invoice_no), 2) AS average_order_value,
    ROUND(SUM(total_amount) / COUNT(DISTINCT customer_id), 2) AS average_customer_spend
FROM cleaned_transactions;

-- 2. Monthly Revenue and Volume Trend
SELECT
    STRFTIME(invoice_date, '%Y-%m') AS year_month,
    COUNT(DISTINCT invoice_no) AS total_invoices,
    COUNT(DISTINCT customer_id) AS active_customers,
    SUM(quantity) AS total_units_sold,
    ROUND(SUM(total_amount), 2) AS monthly_revenue
FROM cleaned_transactions
GROUP BY 1
ORDER BY 1 ASC;

-- 3. Top 20 Customers by Historical Revenue
SELECT
    customer_id,
    num_invoices,
    ROUND(total_revenue, 2) AS total_revenue,
    ROUND(avg_order_value, 2) AS avg_order_value,
    ROUND(recency_days, 1) AS recency_days,
    ROUND(customer_lifespan_days, 1) AS lifespan_days
FROM customer_metrics
ORDER BY total_revenue DESC
LIMIT 20;

-- 4. Segment Distribution and Revenue Concentration
SELECT
    segment,
    COUNT(*) AS customer_count,
    ROUND(SUM(total_revenue), 2) AS segment_revenue,
    ROUND(AVG(avg_order_value), 2) AS avg_aov,
    ROUND(AVG(recency_days), 1) AS avg_recency_days
FROM customer_metrics
GROUP BY segment
ORDER BY segment_revenue DESC;

-- 5. Repeat vs Single-Purchase Customer Breakdown
SELECT
    CASE WHEN num_invoices > 1 THEN 'Repeat Customer' ELSE 'Single-Purchase Customer' END AS customer_type,
    COUNT(*) AS customer_count,
    ROUND(SUM(total_revenue), 2) AS total_revenue,
    ROUND(AVG(total_revenue), 2) AS avg_spend
FROM customer_metrics
GROUP BY 1;

-- 6. High-Value At-Risk Accounts (Dormant > 90 days with above-average spend)
SELECT
    customer_id,
    num_invoices,
    ROUND(total_revenue, 2) AS total_revenue,
    ROUND(recency_days, 1) AS days_since_last_purchase,
    segment
FROM customer_metrics
WHERE recency_days >= 90
ORDER BY total_revenue DESC
LIMIT 50;

-- 7. Pipeline Run History Audit
SELECT
    run_id,
    execution_timestamp,
    dataset_source,
    raw_row_count,
    cleaned_row_count,
    unique_customers,
    status,
    execution_duration_sec
FROM pipeline_metadata
ORDER BY execution_timestamp DESC;
