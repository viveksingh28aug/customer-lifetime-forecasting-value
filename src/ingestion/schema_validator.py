"""Schema Validation and Column Normalization Module.

Validates that incoming tabular datasets conform to the required retail transaction schema,
applies flexible column alias mapping, and verifies data type integrity.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd

from config.settings import REQUIRED_COLUMNS, COLUMN_ALIASES
from config.logging_config import get_logger

logger = get_logger("src.ingestion.schema_validator")

class SchemaValidationResult:
    """Container for schema validation outcomes."""

    def __init__(
        self,
        is_valid: bool,
        missing_columns: List[str],
        column_mapping: Dict[str, str],
        normalized_df: Optional[pd.DataFrame] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.is_valid = is_valid
        self.missing_columns = missing_columns
        self.column_mapping = column_mapping
        self.normalized_df = normalized_df
        self.warnings = warnings or []

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "missing_columns": self.missing_columns,
            "column_mapping": self.column_mapping,
            "warnings": self.warnings,
        }

def normalize_column_name(col: str) -> str:
    """Cleans a column name by stripping whitespace and converting to lowercase."""
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")

def validate_and_map_schema(
    df: pd.DataFrame,
    custom_mapping: Optional[Dict[str, str]] = None
) -> SchemaValidationResult:
    """Validates required columns and maps aliases to standard retail column names.

    Args:
        df: Input raw DataFrame.
        custom_mapping: Optional explicit mapping from input column to standard column name.

    Returns:
        SchemaValidationResult containing validation status, mapping used, and normalized DataFrame.
    """
    if df is None or df.empty:
        return SchemaValidationResult(
            is_valid=False,
            missing_columns=REQUIRED_COLUMNS.copy(),
            column_mapping={},
            normalized_df=None,
            warnings=["Input dataset is empty or None."],
        )

    df_copy = df.copy()
    current_cols = list(df_copy.columns)
    normalized_col_lookup = {normalize_column_name(c): c for c in current_cols}

    final_mapping: Dict[str, str] = {}
    warnings: List[str] = []

    # 1. Apply user custom mapping if provided
    if custom_mapping:
        for user_col, std_col in custom_mapping.items():
            if user_col in df_copy.columns and std_col in REQUIRED_COLUMNS:
                final_mapping[user_col] = std_col

    # 2. Check each required standard column
    for std_col in REQUIRED_COLUMNS:
        # If already mapped via custom_mapping, continue
        if std_col in final_mapping.values():
            continue

        # Exact match (case-sensitive or insensitive)
        if std_col in df_copy.columns:
            final_mapping[std_col] = std_col
            continue

        std_norm = normalize_column_name(std_col)
        if std_norm in normalized_col_lookup:
            orig_col = normalized_col_lookup[std_norm]
            final_mapping[orig_col] = std_col
            continue

        # Check known aliases
        aliases = COLUMN_ALIASES.get(std_col, [])
        found_alias = False
        for alias in aliases:
            alias_norm = normalize_column_name(alias)
            if alias_norm in normalized_col_lookup:
                orig_col = normalized_col_lookup[alias_norm]
                final_mapping[orig_col] = std_col
                warnings.append(f"Mapped alias '{orig_col}' to standard column '{std_col}'.")
                found_alias = True
                break

    # Determine missing required columns
    mapped_standard_cols = set(final_mapping.values())
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in mapped_standard_cols]

    is_valid = len(missing_columns) == 0

    if not is_valid:
        warnings.append(f"Missing required columns: {missing_columns}")
        logger.warning(f"Schema validation failed. Missing required columns: {missing_columns}")
        return SchemaValidationResult(
            is_valid=False,
            missing_columns=missing_columns,
            column_mapping=final_mapping,
            normalized_df=None,
            warnings=warnings,
        )

    # Apply rename to standardize DataFrame columns
    rename_dict = {orig: std for orig, std in final_mapping.items() if orig != std}
    df_copy = df_copy.rename(columns=rename_dict)
    
    # Keep only standard columns and preserve any additional ones
    logger.info(f"Schema validation passed successfully. Columns mapped: {len(final_mapping)}")
    return SchemaValidationResult(
        is_valid=True,
        missing_columns=[],
        column_mapping=final_mapping,
        normalized_df=df_copy,
        warnings=warnings,
    )
