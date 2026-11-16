"""
Data acquisition, inspection, and multi-file loading module for CIC-IDS2017.
Supports efficient chunked loading, memory downcasting, and software verification fixtures.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

from src.config import config, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Canonical CIC-IDS2017 attack classes
CANONICAL_ATTACK_CLASSES = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Infiltration",
    "Web Attack - Sql Injection",
    "Heartbleed",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strips leading/trailing whitespace and normalizes column headers.
    CIC-IDS2017 is known for inconsistent leading spaces (e.g. ' Destination Port').
    """
    cleaned_cols = [str(col).strip() for col in df.columns]
    df.columns = cleaned_cols
    return df


def check_raw_dataset_status(raw_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Checks the status of raw CIC-IDS2017 CSV files in the raw directory.
    Returns status metadata including file count, sizes, and whether real raw data exists.
    """
    dir_path = Path(raw_dir) if raw_dir else config.raw_data_dir
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(dir_path.glob("*.csv"))
    total_size_mb = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)

    return {
        "directory": str(dir_path),
        "raw_files_found": [f.name for f in csv_files],
        "file_count": len(csv_files),
        "total_size_mb": round(total_size_mb, 2),
        "is_available": len(csv_files) > 0,
        "required_official_files": config.get("dataset.raw_files", []),
    }


def generate_sample_fixture(
    output_path: Optional[Union[str, Path]] = None,
    n_records: int = 5000,
    random_state: int = 42,
) -> Path:
    """
    Generates a schema-accurate CIC-IDS2017 test fixture for software verification
    and unit tests when the complete multi-gigabyte raw dataset has not yet been downloaded.
    Includes benign and attack traffic with realistic distributions, plus deliberate
    edge cases (NaNs, Infs, duplicate records) to test the cleaning pipeline.
    """
    rng = np.random.RandomState(random_state)
    target_path = Path(output_path) if output_path else config.sample_data_dir / "cicids2017_sample.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Label distribution: ~70% BENIGN, ~30% Attacks
    labels_pool = [
        "BENIGN",
        "DoS Hulk",
        "PortScan",
        "DDoS",
        "DoS GoldenEye",
        "FTP-Patator",
        "SSH-Patator",
        "DoS slowloris",
        "Bot",
        "Web Attack - Brute Force",
    ]
    probabilities = [0.70, 0.08, 0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01]
    assigned_labels = rng.choice(labels_pool, size=n_records, p=probabilities)

    data: Dict[str, Any] = {
        "Destination Port": rng.choice([80, 443, 22, 21, 8080, 53, 445, 139, 3389], size=n_records),
        "Flow Duration": rng.exponential(scale=1000000, size=n_records).clip(1, 120000000).astype(int),
        "Total Fwd Packets": rng.poisson(lam=10, size=n_records).clip(1, 10000),
        "Total Backward Packets": rng.poisson(lam=12, size=n_records).clip(0, 10000),
        "Total Length of Fwd Packets": rng.exponential(scale=1500, size=n_records).clip(0, 500000),
        "Total Length of Bwd Packets": rng.exponential(scale=3000, size=n_records).clip(0, 1000000),
        "Fwd Packet Length Max": rng.uniform(40, 1460, size=n_records),
        "Fwd Packet Length Min": rng.uniform(0, 60, size=n_records),
        "Fwd Packet Length Mean": rng.uniform(40, 750, size=n_records),
        "Fwd Packet Length Std": rng.uniform(0, 300, size=n_records),
        "Bwd Packet Length Max": rng.uniform(40, 1460, size=n_records),
        "Bwd Packet Length Min": rng.uniform(0, 60, size=n_records),
        "Bwd Packet Length Mean": rng.uniform(40, 900, size=n_records),
        "Bwd Packet Length Std": rng.uniform(0, 400, size=n_records),
        "Flow Bytes/s": rng.exponential(scale=50000, size=n_records).clip(0, 10000000),
        "Flow Packets/s": rng.exponential(scale=100, size=n_records).clip(0, 500000),
        "Flow IAT Mean": rng.exponential(scale=50000, size=n_records),
        "Flow IAT Std": rng.exponential(scale=70000, size=n_records),
        "Flow IAT Max": rng.exponential(scale=200000, size=n_records),
        "Flow IAT Min": rng.uniform(0, 100, size=n_records),
        "Fwd IAT Total": rng.exponential(scale=500000, size=n_records),
        "Fwd IAT Mean": rng.exponential(scale=50000, size=n_records),
        "Fwd IAT Std": rng.exponential(scale=60000, size=n_records),
        "Fwd IAT Max": rng.exponential(scale=180000, size=n_records),
        "Fwd IAT Min": rng.uniform(0, 80, size=n_records),
        "Bwd IAT Total": rng.exponential(scale=500000, size=n_records),
        "Bwd IAT Mean": rng.exponential(scale=45000, size=n_records),
        "Bwd IAT Std": rng.exponential(scale=55000, size=n_records),
        "Bwd IAT Max": rng.exponential(scale=170000, size=n_records),
        "Bwd IAT Min": rng.uniform(0, 80, size=n_records),
        "Fwd PSH Flags": rng.choice([0, 1], size=n_records, p=[0.85, 0.15]),
        "Bwd PSH Flags": np.zeros(n_records, dtype=int),
        "Fwd URG Flags": np.zeros(n_records, dtype=int),
        "Bwd URG Flags": np.zeros(n_records, dtype=int),
        "Fwd Header Length": rng.choice([20, 32, 40, 60], size=n_records) * rng.poisson(lam=10, size=n_records),
        "Bwd Header Length": rng.choice([20, 32, 40, 60], size=n_records) * rng.poisson(lam=12, size=n_records),
        "Fwd Packets/s": rng.exponential(scale=50, size=n_records),
        "Bwd Packets/s": rng.exponential(scale=50, size=n_records),
        "Min Packet Length": rng.uniform(0, 60, size=n_records),
        "Max Packet Length": rng.uniform(100, 1500, size=n_records),
        "Packet Length Mean": rng.uniform(50, 800, size=n_records),
        "Packet Length Std": rng.uniform(10, 400, size=n_records),
        "Packet Length Variance": rng.uniform(100, 160000, size=n_records),
        "FIN Flag Count": rng.choice([0, 1], size=n_records, p=[0.92, 0.08]),
        "SYN Flag Count": rng.choice([0, 1], size=n_records, p=[0.88, 0.12]),
        "RST Flag Count": rng.choice([0, 1], size=n_records, p=[0.97, 0.03]),
        "PSH Flag Count": rng.choice([0, 1], size=n_records, p=[0.70, 0.30]),
        "ACK Flag Count": rng.choice([0, 1], size=n_records, p=[0.35, 0.65]),
        "URG Flag Count": rng.choice([0, 1], size=n_records, p=[0.95, 0.05]),
        "CWE Flag Count": np.zeros(n_records, dtype=int),
        "ECE Flag Count": rng.choice([0, 1], size=n_records, p=[0.98, 0.02]),
        "Down/Up Ratio": rng.choice([0, 1, 2, 3], size=n_records, p=[0.4, 0.45, 0.1, 0.05]),
        "Average Packet Size": rng.uniform(60, 900, size=n_records),
        "Avg Fwd Segment Size": rng.uniform(40, 750, size=n_records),
        "Avg Bwd Segment Size": rng.uniform(40, 900, size=n_records),
        "Fwd Header Length.1": rng.choice([20, 32, 40], size=n_records) * rng.poisson(lam=10, size=n_records),
        "Fwd Avg Bytes/Bulk": np.zeros(n_records),
        "Fwd Avg Packets/Bulk": np.zeros(n_records),
        "Fwd Avg Bulk Rate": np.zeros(n_records),
        "Bwd Avg Bytes/Bulk": np.zeros(n_records),
        "Bwd Avg Packets/Bulk": np.zeros(n_records),
        "Bwd Avg Bulk Rate": np.zeros(n_records),
        "Subflow Fwd Packets": rng.poisson(lam=10, size=n_records),
        "Subflow Fwd Bytes": rng.exponential(scale=1500, size=n_records),
        "Subflow Bwd Packets": rng.poisson(lam=12, size=n_records),
        "Subflow Bwd Bytes": rng.exponential(scale=3000, size=n_records),
        "Init_Win_bytes_forward": rng.choice([29200, 65535, 8192, 26883, 14600, -1], size=n_records),
        "Init_Win_bytes_backward": rng.choice([29200, 65535, 8192, 26883, 14600, -1], size=n_records),
        "act_data_pkt_fwd": rng.poisson(lam=5, size=n_records),
        "min_seg_size_forward": rng.choice([20, 32], size=n_records),
        "Active Mean": rng.exponential(scale=10000, size=n_records),
        "Active Std": rng.exponential(scale=5000, size=n_records),
        "Active Max": rng.exponential(scale=20000, size=n_records),
        "Active Min": rng.uniform(0, 1000, size=n_records),
        "Idle Mean": rng.exponential(scale=100000, size=n_records),
        "Idle Std": rng.exponential(scale=20000, size=n_records),
        "Idle Max": rng.exponential(scale=250000, size=n_records),
        "Idle Min": rng.uniform(0, 5000, size=n_records),
        "Label": assigned_labels,
    }

    df = pd.DataFrame(data)

    # Intentionally inject realistic anomalies for cleaning pipeline verification:
    # 1. Infinite values in Flow Bytes/s and Flow Packets/s (classic CIC-IDS2017 artifact)
    inf_indices = rng.choice(n_records, size=max(1, int(n_records * 0.005)), replace=False)
    df.loc[inf_indices, "Flow Bytes/s"] = np.inf
    df.loc[inf_indices[: len(inf_indices) // 2], "Flow Packets/s"] = np.inf

    # 2. Missing values (NaNs) in packet statistics
    nan_indices = rng.choice(n_records, size=max(1, int(n_records * 0.004)), replace=False)
    df.loc[nan_indices, "Flow Bytes/s"] = np.nan
    df.loc[nan_indices, "Fwd Packet Length Std"] = np.nan

    # 3. Duplicate records (~1% duplicates)
    dup_indices = rng.choice(n_records, size=int(n_records * 0.01), replace=False)
    duplicates = df.iloc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    df.to_csv(target_path, index=False)
    logger.info(f"Generated software verification sample fixture with {len(df)} rows at {target_path}")
    return target_path


def load_dataset(
    data_path: Optional[Union[str, Path]] = None,
    sample_fraction: Optional[float] = None,
    max_rows: Optional[int] = None,
    random_state: int = 42,
    use_sample_if_empty: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads one or multiple CIC-IDS2017 CSV files from data/raw or a specified path.
    If data/raw is empty and use_sample_if_empty is True, automatically generates/loads
    the software testing sample fixture with explicit metadata tracking.

    Returns:
        (DataFrame, metadata_dict)
    """
    metadata: Dict[str, Any] = {
        "source": "unknown",
        "is_sample": False,
        "files_loaded": [],
        "total_records_read": 0,
        "sampled_records": 0,
    }

    # Case 1: Specific single path provided
    if data_path:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Specified dataset path does not exist: {path}")
        logger.info(f"Loading single dataset file: {path}")
        df = pd.read_csv(path, low_memory=False)
        df = clean_column_names(df)
        metadata["source"] = str(path)
        metadata["files_loaded"] = [path.name]
        metadata["is_sample"] = "sample" in str(path).lower()

    # Case 2: Load from data/raw/
    else:
        raw_status = check_raw_dataset_status(config.raw_data_dir)
        raw_files = [config.raw_data_dir / f for f in raw_status["raw_files_found"]]

        if raw_files:
            logger.info(f"Found {len(raw_files)} raw files in {config.raw_data_dir}. Loading...")
            dfs = []
            for rf in raw_files:
                logger.info(f"Reading {rf.name}...")
                chunk = pd.read_csv(rf, low_memory=False)
                chunk = clean_column_names(chunk)
                if sample_fraction and 0.0 < sample_fraction < 1.0:
                    chunk = chunk.sample(frac=sample_fraction, random_state=random_state)
                dfs.append(chunk)

            df = pd.concat(dfs, ignore_index=True)
            metadata["source"] = "data/raw"
            metadata["files_loaded"] = [f.name for f in raw_files]
            metadata["is_sample"] = False

        elif use_sample_if_empty:
            sample_file = config.sample_data_dir / "cicids2017_sample.csv"
            if not sample_file.exists():
                logger.warning("No raw datasets found. Generating verified software testing fixture...")
                generate_sample_fixture(sample_file, n_records=5000, random_state=random_state)

            logger.info(f"Loading software verification sample from {sample_file}")
            df = pd.read_csv(sample_file, low_memory=False)
            df = clean_column_names(df)
            metadata["source"] = "data/sample (Software Verification Fixture)"
            metadata["files_loaded"] = [sample_file.name]
            metadata["is_sample"] = True

        else:
            raise FileNotFoundError(
                f"No CSV datasets found in {config.raw_data_dir}. Please download CIC-IDS2017 "
                "or enable software verification mode."
            )

    # Optional global row limit for rapid prototyping
    metadata["total_records_read"] = len(df)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        metadata["sampled_records"] = len(df)
    else:
        metadata["sampled_records"] = len(df)

    return df, metadata


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive dataset inspection for Data Understanding.
    Calculates missing values, duplicates, infinite values, memory usage,
    and class distributions.
    """
    total_rows = len(df)
    total_cols = len(df.columns)

    # Missing values
    missing_series = df.isnull().sum()
    total_missing = int(missing_series.sum())
    cols_with_missing = {
        col: int(cnt) for col, cnt in missing_series[missing_series > 0].items()
    }

    # Infinite values
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    total_infs = 0
    for col in num_cols:
        count_inf = int(np.isinf(df[col]).sum())
        if count_inf > 0:
            inf_counts[col] = count_inf
            total_infs += count_inf

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # Memory usage
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    # Target label analysis
    target_col = config.target_column
    if target_col in df.columns:
        label_counts = df[target_col].value_counts().to_dict()
        unique_labels = list(label_counts.keys())
        benign_count = int(label_counts.get(config.benign_label, 0))
        attack_count = total_rows - benign_count
    else:
        label_counts = {}
        unique_labels = []
        benign_count = 0
        attack_count = 0

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "columns": list(df.columns),
        "dtypes_distribution": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        "total_missing_values": total_missing,
        "columns_with_missing": cols_with_missing,
        "total_infinite_values": total_infs,
        "columns_with_infinite": inf_counts,
        "duplicate_rows": duplicate_count,
        "memory_usage_mb": memory_mb,
        "unique_labels": unique_labels,
        "label_counts": label_counts,
        "benign_count": benign_count,
        "attack_count": attack_count,
        "attack_percentage": round((attack_count / total_rows * 100), 2) if total_rows > 0 else 0.0,
        "is_clean": (total_missing == 0 and total_infs == 0 and duplicate_count == 0),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    status = check_raw_dataset_status()
    print("Raw Dataset Status:", status)
    df, meta = load_dataset(use_sample_if_empty=True)
    summary = inspect_dataset(df)
    print("Dataset Summary:")
    for k, v in summary.items():
        if k != "columns":
            print(f"  {k}: {v}")

