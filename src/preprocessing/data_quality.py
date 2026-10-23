"""Data Quality Assessment and Reporting Module.

Calculates data quality metrics, invalid record frequencies, integrity flags,
and generates structured reports for auditability and compliance.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd

from config.logging_config import get_logger

logger = get_logger("src.preprocessing.data_quality")

@dataclass
class DataQualityReport:
    """Standardized audit report of data quality and cleaning exclusions."""

    run_id: str
    total_raw_rows: int
    total_valid_rows: int
    duplicate_rows: int
    missing_customer_ids: int
    invalid_dates: int
    return_records: int
    invalid_unit_prices: int
    excluded_rows: int
    final_cleaned_rows: int
    cleaning_policy: str
    data_quality_score: float
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts metrics into a clean display DataFrame."""
        records = [
            {"Metric": "Total Raw Transactions", "Count / Value": self.total_raw_rows, "Description": "Total records ingested"},
            {"Metric": "Duplicate Records Removed", "Count / Value": self.duplicate_rows, "Description": "Identical transactions deduplicated"},
            {"Metric": "Missing Customer Identifiers", "Count / Value": self.missing_customer_ids, "Description": "Guest checkouts without CustomerID"},
            {"Metric": "Invalid Date Formats", "Count / Value": self.invalid_dates, "Description": "Unparseable timestamp records"},
            {"Metric": "Cancellation / Return Records", "Count / Value": self.return_records, "Description": "Invoices with negative quantity or 'C' prefix"},
            {"Metric": "Invalid Unit Prices", "Count / Value": self.invalid_unit_prices, "Description": "Transactions with price <= 0.00"},
            {"Metric": "Total Excluded Records", "Count / Value": self.excluded_rows, "Description": "Rows excluded from primary sales modeling"},
            {"Metric": "Final Cleaned Transactions", "Count / Value": self.final_cleaned_rows, "Description": "High-integrity transactions used in modeling"},
            {"Metric": "Data Quality Health Score", "Count / Value": f"{self.data_quality_score:.1f}%", "Description": "Ratio of clean sales to raw records"},
        ]
        return pd.DataFrame(records)

    def to_sql_record(self) -> Dict[str, Any]:
        """Maps to database table schema."""
        return {
            "run_id": self.run_id,
            "total_raw_rows": self.total_raw_rows,
            "total_valid_rows": self.total_valid_rows,
            "duplicate_rows": self.duplicate_rows,
            "missing_customer_ids": self.missing_customer_ids,
            "invalid_dates": self.invalid_dates,
            "return_records": self.return_records,
            "excluded_rows": self.excluded_rows,
            "final_cleaned_rows": self.final_cleaned_rows,
            "generated_timestamp": self.generated_at,
        }
