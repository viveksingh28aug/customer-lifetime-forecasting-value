-- Analytical Schema Definition for CLV & Revenue Forecasting Engine
-- Compatible with DuckDB, SQLite, and PostgreSQL

CREATE TABLE IF NOT EXISTS pipeline_metadata (
    run_id VARCHAR(64) PRIMARY KEY,
    execution_timestamp TIMESTAMP,
    dataset_source VARCHAR(255),
    raw_row_count INTEGER,
    cleaned_row_count INTEGER,
    unique_customers INTEGER,
    status VARCHAR(32),
    execution_duration_sec DOUBLE,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS data_quality_reports (
    run_id VARCHAR(64) PRIMARY KEY,
    total_raw_rows INTEGER,
    total_valid_rows INTEGER,
    duplicate_rows INTEGER,
    missing_customer_ids INTEGER,
    invalid_dates INTEGER,
    return_records INTEGER,
    excluded_rows INTEGER,
    final_cleaned_rows INTEGER,
    generated_timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_transactions (
    invoice_no VARCHAR(64),
    stock_code VARCHAR(64),
    description TEXT,
    quantity INTEGER,
    invoice_date TIMESTAMP,
    unit_price DOUBLE,
    customer_id VARCHAR(64),
    country VARCHAR(100),
    run_id VARCHAR(64),
    ingestion_timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cleaned_transactions (
    invoice_no VARCHAR(64),
    stock_code VARCHAR(64),
    description TEXT,
    quantity INTEGER,
    invoice_date TIMESTAMP,
    unit_price DOUBLE,
    customer_id VARCHAR(64),
    country VARCHAR(100),
    total_amount DOUBLE,
    is_return BOOLEAN,
    run_id VARCHAR(64),
    cleaned_timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customer_metrics (
    customer_id VARCHAR(64) PRIMARY KEY,
    first_purchase_date TIMESTAMP,
    last_purchase_date TIMESTAMP,
    num_invoices INTEGER,
    total_quantity INTEGER,
    total_revenue DOUBLE,
    avg_order_value DOUBLE,
    avg_revenue_per_purchase DOUBLE,
    recency_days DOUBLE,
    purchase_frequency DOUBLE,
    customer_lifespan_days DOUBLE,
    avg_interpurchase_time DOUBLE,
    run_id VARCHAR(64),
    calculated_timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rfm_segments (
    customer_id VARCHAR(64) PRIMARY KEY,
    recency_score INTEGER,
    frequency_score INTEGER,
    monetary_score INTEGER,
    rfm_score VARCHAR(16),
    segment VARCHAR(64),
    run_id VARCHAR(64),
    assigned_timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clv_predictions (
    customer_id VARCHAR(64),
    model_type VARCHAR(64),
    horizon_months INTEGER,
    predicted_transactions DOUBLE,
    predicted_spend_per_tx DOUBLE,
    predicted_clv DOUBLE,
    gross_margin DOUBLE,
    discount_rate DOUBLE,
    run_id VARCHAR(64),
    prediction_timestamp TIMESTAMP,
    PRIMARY KEY (customer_id, model_type, horizon_months)
);

CREATE TABLE IF NOT EXISTS revenue_forecasts (
    period_date TIMESTAMP,
    model_name VARCHAR(64),
    forecast_horizon INTEGER,
    predicted_revenue DOUBLE,
    lower_bound DOUBLE,
    upper_bound DOUBLE,
    is_test_actual BOOLEAN,
    actual_revenue DOUBLE,
    run_id VARCHAR(64),
    created_timestamp TIMESTAMP,
    PRIMARY KEY (period_date, model_name, forecast_horizon)
);
