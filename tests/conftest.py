"""Pytest Shared Fixtures for CLV & Revenue Forecasting Engine."""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def sample_clean_transactions() -> pd.DataFrame:
    """Fixture providing clean, valid retail transactions for 4 customers across 4 months."""
    return pd.DataFrame([
        # Customer 101: 3 orders across Jan, Feb, Mar (Repeat VIP)
        {"InvoiceNo": "500001", "StockCode": "ITEM_A", "Description": "Product A", "Quantity": 2, "InvoiceDate": "2024-01-10 10:00:00", "UnitPrice": 10.0, "CustomerID": "101", "Country": "UK"},
        {"InvoiceNo": "500002", "StockCode": "ITEM_B", "Description": "Product B", "Quantity": 1, "InvoiceDate": "2024-02-15 12:00:00", "UnitPrice": 30.0, "CustomerID": "101", "Country": "UK"},
        {"InvoiceNo": "500003", "StockCode": "ITEM_C", "Description": "Product C", "Quantity": 4, "InvoiceDate": "2024-03-20 14:00:00", "UnitPrice": 25.0, "CustomerID": "101", "Country": "UK"},
        
        # Customer 102: 2 orders in Jan, Mar
        {"InvoiceNo": "500004", "StockCode": "ITEM_A", "Description": "Product A", "Quantity": 1, "InvoiceDate": "2024-01-12 11:00:00", "UnitPrice": 10.0, "CustomerID": "102", "Country": "France"},
        {"InvoiceNo": "500005", "StockCode": "ITEM_D", "Description": "Product D", "Quantity": 2, "InvoiceDate": "2024-03-10 15:00:00", "UnitPrice": 50.0, "CustomerID": "102", "Country": "France"},
        
        # Customer 103: 1 single purchase in Jan (Single-purchase buyer)
        {"InvoiceNo": "500006", "StockCode": "ITEM_B", "Description": "Product B", "Quantity": 1, "InvoiceDate": "2024-01-18 09:30:00", "UnitPrice": 30.0, "CustomerID": "103", "Country": "Germany"},
        
        # Customer 104: Recent single purchase in Apr
        {"InvoiceNo": "500007", "StockCode": "ITEM_C", "Description": "Product C", "Quantity": 2, "InvoiceDate": "2024-04-05 16:00:00", "UnitPrice": 25.0, "CustomerID": "104", "Country": "UK"},
    ])

@pytest.fixture
def sample_messy_transactions() -> pd.DataFrame:
    """Fixture containing real-world data quality flaws (duplicates, missing customer, returns, bad dates)."""
    return pd.DataFrame([
        # Valid order
        {"InvoiceNo": "600001", "StockCode": "PROD_1", "Description": "Item 1", "Quantity": 2, "InvoiceDate": "2024-01-05 10:00:00", "UnitPrice": 20.0, "CustomerID": "201", "Country": "UK"},
        # Exact duplicate of the above
        {"InvoiceNo": "600001", "StockCode": "PROD_1", "Description": "Item 1", "Quantity": 2, "InvoiceDate": "2024-01-05 10:00:00", "UnitPrice": 20.0, "CustomerID": "201", "Country": "UK"},
        # Missing CustomerID (Guest checkout)
        {"InvoiceNo": "600002", "StockCode": "PROD_2", "Description": "Item 2", "Quantity": 1, "InvoiceDate": "2024-01-06 11:00:00", "UnitPrice": 15.0, "CustomerID": None, "Country": "UK"},
        # Float-formatted Customer ID (e.g. 202.0)
        {"InvoiceNo": "600003", "StockCode": "PROD_3", "Description": "Item 3", "Quantity": 3, "InvoiceDate": "2024-01-07 12:00:00", "UnitPrice": 10.0, "CustomerID": 202.0, "Country": "Germany"},
        # Cancellation with negative quantity & 'C' prefix
        {"InvoiceNo": "C600004", "StockCode": "PROD_1", "Description": "Item 1 return", "Quantity": -1, "InvoiceDate": "2024-01-08 14:00:00", "UnitPrice": 20.0, "CustomerID": "201", "Country": "UK"},
        # Zero UnitPrice
        {"InvoiceNo": "600005", "StockCode": "PROD_FREE", "Description": "Free Gift", "Quantity": 1, "InvoiceDate": "2024-01-09 15:00:00", "UnitPrice": 0.0, "CustomerID": "203", "Country": "France"},
        # Invalid / Unparseable Date
        {"InvoiceNo": "600006", "StockCode": "PROD_4", "Description": "Bad Date Item", "Quantity": 1, "InvoiceDate": "NOT_A_DATE_VALUE", "UnitPrice": 25.0, "CustomerID": "204", "Country": "UK"},
    ])
