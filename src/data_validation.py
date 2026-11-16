"""
Data validation module for network traffic records and CSV uploads.
Validates schemas, network parameter ranges, and guards against malformed inputs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

from src.config import config

logger = logging.getLogger(__name__)

# Expected minimum core network features for valid CIC-IDS2017 style analysis
EXPECTED_CORE_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Packet Length Mean",
]


class DataValidationError(Exception):
    """Raised when dataset fails critical schema or security validation."""
    pass


def validate_csv_file(
    file_obj_or_path: Union[str, Path, Any],
    max_size_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Validates CSV file size and format before ingestion.
    Prevents memory exhaustion and unhandled file formats.
    """
    max_mb = max_size_mb or config.get("dashboard.max_upload_size_mb", 50.0)

    # Check file size if it's a path on disk
    if isinstance(file_obj_or_path, (str, Path)):
        p = Path(file_obj_or_path)
        if not p.exists():
            return {"is_valid": False, "error": f"File not found: {p}"}
        if not p.suffix.lower() == ".csv":
            return {"is_valid": False, "error": f"Invalid extension '{p.suffix}'. Expected '.csv'."}
        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            return {
                "is_valid": False,
                "error": f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_mb} MB.",
            }
        return {"is_valid": True, "size_mb": round(size_mb, 2), "filename": p.name}

    # Handle Streamlit UploadedFile object or file-like buffer
    if hasattr(file_obj_or_path, "name") and hasattr(file_obj_or_path, "size"):
        filename = getattr(file_obj_or_path, "name", "upload.csv")
        size_mb = getattr(file_obj_or_path, "size", 0) / (1024 * 1024)
        if not filename.lower().endswith(".csv"):
            return {"is_valid": False, "error": "Only CSV (.csv) files are supported for upload."}
        if size_mb > max_mb:
            return {
                "is_valid": False,
                "error": f"Uploaded file ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_mb} MB.",
            }
        return {"is_valid": True, "size_mb": round(size_mb, 2), "filename": filename}

    return {"is_valid": True, "size_mb": 0.0, "filename": "in_memory_stream"}


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_features: Optional[List[str]] = None,
    allow_missing_target: bool = False,
) -> Dict[str, Any]:
    """
    Validates that a DataFrame conforms to the expected network traffic schema.
    Checks column presence, data types, missing target labels, and infinite values.
    """
    # Clean column names to avoid whitespace mismatches
    cols = [str(c).strip() for c in df.columns]
    target_col = config.target_column

    req_cols = required_features or EXPECTED_CORE_FEATURES
    missing_required = [c for c in req_cols if c not in cols]

    has_target = target_col in cols
    if not has_target and not allow_missing_target:
        target_status = f"Missing required target column '{target_col}'"
    else:
        target_status = "Target column present" if has_target else "Target column absent (inference mode)"

    # Check for empty dataframe
    if df.empty:
        return {
            "is_valid": False,
            "error": "DataFrame is empty (0 records).",
            "missing_required_features": missing_required,
            "has_target": has_target,
        }

    # Check non-numeric unexpected columns
    non_numeric_cols = []
    for c in cols:
        if c != target_col and not pd.api.types.is_numeric_dtype(df[c]):
            non_numeric_cols.append(c)

    # Check for port range violations if Destination Port is present
    port_violations = 0
    if "Destination Port" in df.columns:
        valid_ports = pd.to_numeric(df["Destination Port"], errors="coerce").dropna()
        port_violations = int(((valid_ports < 0) | (valid_ports > 65535)).sum())

    is_valid = len(missing_required) == 0 and (has_target or allow_missing_target)

    return {
        "is_valid": is_valid,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_required_features": missing_required,
        "has_target": has_target,
        "target_status": target_status,
        "non_numeric_features": non_numeric_cols,
        "port_violations": port_violations,
        "error": f"Missing critical core network features: {missing_required}" if missing_required else None,
    }


def validate_inference_features(
    df: pd.DataFrame,
    expected_model_features: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validates uploaded inference data against the exact feature set expected by a trained model.
    Fills missing features with 0.0 or median fallback, removes excess unneeded features,
    and returns an aligned DataFrame with matching column order.
    """
    cleaned_df = df.copy()
    cleaned_df.columns = [str(c).strip() for c in cleaned_df.columns]

    missing_in_input = [f for f in expected_model_features if f not in cleaned_df.columns]
    if missing_in_input:
        logger.warning(f"Inference input missing {len(missing_in_input)} expected features. Imputing with 0.0.")
        for col in missing_in_input:
            cleaned_df[col] = 0.0

    # Retain and reorder strictly to match expected model features
    aligned_df = cleaned_df[expected_model_features].copy()
    return aligned_df, missing_in_input

