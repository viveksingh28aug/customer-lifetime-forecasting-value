"""Application Configuration and Settings.

Manages environment variables, directory paths, schema definitions,
cleaning policies, segmentation thresholds, and model parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load optional .env file
load_dotenv()

# Workspace Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure runtime directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, DATABASE_DIR, LOGS_DIR, REPORTS_DIR, ASSETS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Database Settings
DB_BACKEND = os.getenv("DB_BACKEND", "duckdb").lower()
DB_DUCKDB_PATH = BASE_DIR / os.getenv("DB_DUCKDB_PATH", "database/clv_engine.duckdb")
DB_SQLITE_PATH = BASE_DIR / os.getenv("DB_SQLITE_PATH", "database/clv_engine.sqlite")

# PostgreSQL Settings (Optional fallback)
DB_PG_HOST = os.getenv("DB_PG_HOST", "localhost")
DB_PG_PORT = int(os.getenv("DB_PG_PORT", 5432))
DB_PG_NAME = os.getenv("DB_PG_NAME", "clv_engine")
DB_PG_USER = os.getenv("DB_PG_USER", "postgres")
DB_PG_PASSWORD = os.getenv("DB_PG_PASSWORD", "")

# Standard Schema Definition
REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

# Column alias dictionary for flexible ingestion
COLUMN_ALIASES = {
    "InvoiceNo": ["invoiceno", "invoice_no", "invoice_number", "order_id", "transaction_id", "invoice"],
    "StockCode": ["stockcode", "stock_code", "product_code", "sku", "item_id", "item_code"],
    "Description": ["description", "product_name", "item_description", "title", "product_desc"],
    "Quantity": ["quantity", "qty", "units", "volume", "count"],
    "InvoiceDate": ["invoicedate", "invoice_date", "order_date", "transaction_date", "date", "timestamp"],
    "UnitPrice": ["unitprice", "unit_price", "price", "unit_cost", "rate", "item_price"],
    "CustomerID": ["customerid", "customer_id", "client_id", "user_id", "cust_id", "customer"],
    "Country": ["country", "region", "nation", "location", "market"],
}

# Public Dataset URL (UCI Online Retail)
UCI_ONLINE_RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
SYNTHETIC_DATA_FILENAME = "synthetic_online_retail.csv"

# Analytics & CLV Settings
DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "$")
DEFAULT_CLV_HORIZONS = [3, 6, 12]  # Months
DEFAULT_CLV_HORIZON_MONTHS = int(os.getenv("DEFAULT_CLV_HORIZON_MONTHS", 12))
DEFAULT_DISCOUNT_RATE = float(os.getenv("DEFAULT_DISCOUNT_RATE", 0.10))  # Annual 10%
DEFAULT_GROSS_MARGIN = float(os.getenv("DEFAULT_GROSS_MARGIN", 1.0))  # 1.0 = revenue CLV

# RFM Segment Mapping Rules
RFM_SEGMENT_DEFINITIONS = {
    "Champions": "High recency, high frequency, and high monetary spend. Best customers.",
    "Loyal Customers": "Regular buyers with solid frequency and value. Responsive to promotions.",
    "Potential Loyalists": "Recent customers with above-average spend and moderate frequency.",
    "Recent Customers": "New buyers who purchased recently, but low historical frequency.",
    "Customers Needing Attention": "Above average frequency and spend, but haven't purchased recently.",
    "At-Risk Customers": "Purchased often in the past, but haven't purchased in a long time.",
    "Hibernating Customers": "Low recency, low frequency, and low monetary value.",
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / os.getenv("LOG_FILE", "logs/app.log")
