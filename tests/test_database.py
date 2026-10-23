"""Tests for Database Operations (DuckDB and SQLite Fallback)."""

import pandas as pd
import pytest

from database.db_manager import DatabaseManager

def test_duckdb_schema_init_and_query(tmp_path):
    """Test DuckDB database initialization, write, and parameterized query."""
    db_file = tmp_path / "test_duckdb.duckdb"
    db = DatabaseManager(backend="duckdb", db_path=str(db_file))
    db.init_schema()

    # Save DataFrame
    test_df = pd.DataFrame([
        {"customer_id": "C101", "num_invoices": 3, "total_revenue": 150.0},
        {"customer_id": "C102", "num_invoices": 1, "total_revenue": 30.0},
    ])
    db.save_dataframe("customer_metrics", test_df, if_exists="replace")

    # Read back with parameterized query
    res_df = db.query("SELECT * FROM customer_metrics WHERE customer_id = ?", params=["C101"])
    assert len(res_df) == 1
    assert res_df["customer_id"].iloc[0] == "C101"
    assert res_df["total_revenue"].iloc[0] == 150.0

    db.close()

def test_sqlite_fallback_schema_init_and_query(tmp_path):
    """Test SQLite fallback database initialization, write, and query."""
    db_file = tmp_path / "test_sqlite.sqlite"
    db = DatabaseManager(backend="sqlite", db_path=str(db_file))
    db.init_schema()

    test_df = pd.DataFrame([
        {"customer_id": "C201", "num_invoices": 5, "total_revenue": 500.0},
    ])
    db.save_dataframe("customer_metrics", test_df, if_exists="replace")

    res_df = db.query("SELECT * FROM customer_metrics WHERE customer_id = ?", params=["C201"])
    assert len(res_df) == 1
    assert res_df["total_revenue"].iloc[0] == 500.0

    db.close()

def test_database_clean_rerun_replacement(tmp_path):
    """Test rerunning pipeline updates tables without duplication."""
    db_file = tmp_path / "rerun_test.duckdb"
    db = DatabaseManager(backend="duckdb", db_path=str(db_file))
    db.init_schema()

    df_run1 = pd.DataFrame([{"customer_id": "C1", "total_revenue": 100.0}])
    db.save_dataframe("customer_metrics", df_run1, if_exists="replace")

    # Second run replaces
    df_run2 = pd.DataFrame([
        {"customer_id": "C1", "total_revenue": 150.0},
        {"customer_id": "C2", "total_revenue": 200.0},
    ])
    db.save_dataframe("customer_metrics", df_run2, if_exists="replace")

    res = db.query("SELECT * FROM customer_metrics")
    assert len(res) == 2
    assert res[res["customer_id"] == "C1"]["total_revenue"].iloc[0] == 150.0

    db.close()
