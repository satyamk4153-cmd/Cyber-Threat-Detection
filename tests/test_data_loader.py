"""
Unit tests for data acquisition, loading, and inspection module.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    check_raw_dataset_status,
    clean_column_names,
    generate_sample_fixture,
    inspect_dataset,
    load_dataset,
)


def test_clean_column_names():
    df = pd.DataFrame({" Destination Port ": [80], "Flow Duration": [1000], "  Label  ": ["BENIGN"]})
    cleaned = clean_column_names(df)
    assert list(cleaned.columns) == ["Destination Port", "Flow Duration", "Label"]


def test_check_raw_dataset_status(tmp_path):
    status = check_raw_dataset_status(tmp_path)
    assert status["is_available"] is False
    assert status["file_count"] == 0

    # Add dummy CSV
    (tmp_path / "test.csv").write_text("a,b\n1,2")
    status2 = check_raw_dataset_status(tmp_path)
    assert status2["is_available"] is True
    assert status2["file_count"] == 1


def test_generate_sample_fixture(tmp_path):
    sample_file = tmp_path / "sample.csv"
    generate_sample_fixture(sample_file, n_records=500, random_state=42)
    assert sample_file.exists()

    df = pd.read_csv(sample_file)
    assert len(df) >= 500
    assert "Destination Port" in df.columns
    assert "Label" in df.columns
    assert "BENIGN" in df["Label"].values


def test_load_dataset_sample():
    df, meta = load_dataset(use_sample_if_empty=True)
    assert not df.empty
    assert "Label" in df.columns
    assert meta["is_sample"] is True or meta["source"] is not None


def test_inspect_dataset():
    df = pd.DataFrame({
        "Destination Port": [80, 443, 80],
        "Flow Duration": [1000, 2000, 1000],
        "Flow Bytes/s": [100.0, np.inf, 100.0],
        "Label": ["BENIGN", "DoS Hulk", "BENIGN"],
    })
    summary = inspect_dataset(df)
    assert summary["total_rows"] == 3
    assert summary["total_columns"] == 4
    assert summary["total_infinite_values"] == 1
    assert summary["benign_count"] == 2
    assert summary["attack_count"] == 1
    assert summary["duplicate_rows"] == 1

