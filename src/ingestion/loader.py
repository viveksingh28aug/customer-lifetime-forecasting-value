"""Dataset Ingestion and Source Management.

Handles checking local files, downloading public retail datasets (UCI Online Retail),
loading uploaded files (CSV/XLSX), synthetic dataset fallbacks, and dataset provenance tracking.
"""

from datetime import datetime
import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import zipfile
import pandas as pd
import requests

from config.settings import (
    RAW_DATA_DIR,
    SYNTHETIC_DATA_DIR,
    SYNTHETIC_DATA_FILENAME,
    UCI_ONLINE_RETAIL_URL,
)
from config.logging_config import get_logger

logger = get_logger("src.ingestion.loader")

class DatasetMetadata:
    """Stores provenance, source label, and ingestion metadata."""

    def __init__(
        self,
        source_name: str,
        is_synthetic: bool,
        file_path: Optional[str] = None,
        row_count: int = 0,
        ingested_at: Optional[datetime] = None,
        file_size_bytes: int = 0,
    ):
        self.source_name = source_name
        self.is_synthetic = is_synthetic
        self.file_path = file_path
        self.row_count = row_count
        self.ingested_at = ingested_at or datetime.now()
        self.file_size_bytes = file_size_bytes

    def to_dict(self):
        return {
            "source_name": self.source_name,
            "is_synthetic": self.is_synthetic,
            "file_path": str(self.file_path) if self.file_path else None,
            "row_count": self.row_count,
            "ingested_at": self.ingested_at.strftime("%Y-%m-%d %H:%M:%S"),
            "file_size_bytes": self.file_size_bytes,
        }

def get_uci_download_instructions() -> str:
    """Returns manual download instructions for the UCI Online Retail dataset."""
    return (
        "### UCI Online Retail Dataset Instructions\n\n"
        "1. Visit the official UCI Machine Learning Repository:\n"
        "   https://archive.ics.uci.edu/dataset/352/online+retail\n"
        "2. Download the `Online Retail.xlsx` or `.zip` file.\n"
        "3. Place the file inside `data/raw/Online Retail.xlsx` or upload it directly via the dashboard.\n"
        "4. The pipeline will automatically detect and process the file.\n"
    )

def download_uci_dataset(target_dir: Optional[Path] = None, timeout: int = 30) -> Optional[Path]:
    """Attempts automated download of the UCI Online Retail archive.

    Args:
        target_dir: Directory where the dataset will be saved.
        timeout: HTTP request timeout in seconds.

    Returns:
        Path to extracted or downloaded file, or None if download fails.
    """
    target_dir = target_dir or RAW_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_target = target_dir / "online_retail.zip"

    logger.info(f"Attempting to download UCI Online Retail dataset from {UCI_ONLINE_RETAIL_URL}...")
    try:
        response = requests.get(UCI_ONLINE_RETAIL_URL, timeout=timeout, stream=True)
        if response.status_code == 200:
            with open(zip_target, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("UCI Online Retail zip file downloaded successfully.")

            # Unzip if valid zip
            try:
                with zipfile.ZipFile(zip_target, "r") as zip_ref:
                    zip_ref.extractall(target_dir)
                    logger.info(f"Extracted zip archive to {target_dir}")
                
                # Check for xlsx or csv extracted
                for f in target_dir.glob("*.xlsx"):
                    return f
                for f in target_dir.glob("*.csv"):
                    return f
                return zip_target
            except Exception as ze:
                logger.warning(f"Could not extract zip: {ze}")
                return zip_target
        else:
            logger.warning(f"UCI download failed with HTTP status {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Automated download failed (network unavailable or timed out): {e}")
        return None

def find_local_dataset() -> Tuple[Optional[Path], bool]:
    """Scans for available local datasets in data/raw and data/synthetic.

    Returns:
        Tuple of (File Path or None, is_synthetic flag)
    """
    # 1. Check raw directory for user provided or downloaded data
    for ext in ["*.xlsx", "*.xls", "*.csv"]:
        raw_files = list(RAW_DATA_DIR.glob(ext))
        if raw_files:
            return raw_files[0], False

    # 2. Check synthetic directory
    synthetic_file = SYNTHETIC_DATA_DIR / SYNTHETIC_DATA_FILENAME
    if synthetic_file.exists():
        return synthetic_file, True

    return None, False

def load_dataset(
    source: Union[str, Path, io.BytesIO],
    file_type: Optional[str] = None,
    is_synthetic: bool = False,
    source_name: Optional[str] = None
) -> Tuple[pd.DataFrame, DatasetMetadata]:
    """Loads a transaction dataset from file path, buffer, or upload.

    Args:
        source: File path string/Path or in-memory BytesIO buffer.
        file_type: 'csv' or 'excel'. Inferred from extension if not provided.
        is_synthetic: Explicitly tags if the data is synthetic.
        source_name: Human-readable label for the dataset.

    Returns:
        Tuple of (Loaded DataFrame, DatasetMetadata)
    """
    file_path_str = None
    file_size = 0

    if isinstance(source, (str, Path)):
        p = Path(source)
        file_path_str = str(p)
        file_size = p.stat().st_size if p.exists() else 0
        ext = p.suffix.lower()
        if not file_type:
            file_type = "excel" if ext in [".xlsx", ".xls"] else "csv"
        
        name = source_name or p.name
        logger.info(f"Loading dataset from file: {p} (format={file_type})")
        if file_type == "excel":
            df = pd.read_excel(p)
        else:
            # Try utf-8 first, then latin1 fallback (common in retail datasets)
            try:
                df = pd.read_csv(p, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(p, encoding="latin1")

    else:
        # File-like buffer (e.g. from Streamlit UploadedFile)
        name = source_name or "Uploaded Dataset"
        buffer = source
        if hasattr(buffer, "size"):
            file_size = buffer.size
        elif hasattr(buffer, "getbuffer"):
            file_size = len(buffer.getbuffer())

        logger.info(f"Loading dataset from in-memory buffer (format={file_type})")
        if file_type == "excel":
            df = pd.read_excel(buffer)
        else:
            try:
                df = pd.read_csv(buffer, encoding="utf-8")
            except UnicodeDecodeError:
                buffer.seek(0)
                df = pd.read_csv(buffer, encoding="latin1")

    metadata = DatasetMetadata(
        source_name=name,
        is_synthetic=is_synthetic,
        file_path=file_path_str,
        row_count=len(df),
        file_size_bytes=file_size,
    )
    logger.info(f"Loaded {len(df)} rows from {name}. Is Synthetic: {is_synthetic}")
    return df, metadata
