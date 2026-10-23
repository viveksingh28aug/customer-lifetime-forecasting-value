"""Tests for Dataset Ingestion and Schema Validation."""

import io
import pandas as pd
import pytest

from src.ingestion.loader import load_dataset, DatasetMetadata
from src.ingestion.schema_validator import validate_and_map_schema

def test_load_dataset_from_csv(tmp_path):
    """Test loading data from a CSV file path."""
    csv_file = tmp_path / "test_transactions.csv"
    sample_df = pd.DataFrame({
        "InvoiceNo": ["1001", "1002"],
        "StockCode": ["A", "B"],
        "Description": ["Item A", "Item B"],
        "Quantity": [1, 2],
        "InvoiceDate": ["2024-01-01", "2024-01-02"],
        "UnitPrice": [10.5, 20.0],
        "CustomerID": ["C1", "C2"],
        "Country": ["UK", "UK"],
    })
    sample_df.to_csv(csv_file, index=False)

    df_loaded, metadata = load_dataset(csv_file, is_synthetic=True, source_name="Test CSV")
    assert len(df_loaded) == 2
    assert metadata.source_name == "Test CSV"
    assert metadata.is_synthetic is True
    assert metadata.row_count == 2

def test_load_dataset_from_buffer():
    """Test loading data from an in-memory BytesIO buffer."""
    csv_data = b"InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n101,A,Item,1,2024-01-01,5.0,C1,UK\n"
    buffer = io.BytesIO(csv_data)

    df_loaded, metadata = load_dataset(buffer, file_type="csv", source_name="Buffer Test")
    assert len(df_loaded) == 1
    assert "InvoiceNo" in df_loaded.columns
    assert metadata.source_name == "Buffer Test"

def test_schema_validator_exact_match(sample_clean_transactions):
    """Test schema validation succeeds when exact standard column names are present."""
    result = validate_and_map_schema(sample_clean_transactions)
    assert result.is_valid is True
    assert len(result.missing_columns) == 0
    assert result.normalized_df is not None

def test_schema_validator_alias_mapping():
    """Test schema validation auto-maps known column aliases (e.g. order_id -> InvoiceNo)."""
    aliased_df = pd.DataFrame({
        "order_id": ["101"],
        "sku": ["SKU1"],
        "product_name": ["Product 1"],
        "qty": [2],
        "order_date": ["2024-01-01"],
        "price": [15.0],
        "customer_id": ["C100"],
        "location": ["Germany"],
    })
    result = validate_and_map_schema(aliased_df)
    assert result.is_valid is True
    assert len(result.missing_columns) == 0
    assert "InvoiceNo" in result.normalized_df.columns
    assert "StockCode" in result.normalized_df.columns
    assert "Quantity" in result.normalized_df.columns
    assert "UnitPrice" in result.normalized_df.columns

def test_schema_validator_missing_column():
    """Test schema validation fails when a mandatory column is absent."""
    incomplete_df = pd.DataFrame({
        "InvoiceNo": ["101"],
        "StockCode": ["A"],
        "Description": ["Item"],
        # Missing Quantity, UnitPrice, CustomerID, InvoiceDate, Country
    })
    result = validate_and_map_schema(incomplete_df)
    assert result.is_valid is False
    assert "Quantity" in result.missing_columns
    assert "UnitPrice" in result.missing_columns

def test_schema_validator_empty_dataset():
    """Test schema validation with empty DataFrame."""
    empty_df = pd.DataFrame()
    result = validate_and_map_schema(empty_df)
    assert result.is_valid is False
    assert len(result.missing_columns) > 0
