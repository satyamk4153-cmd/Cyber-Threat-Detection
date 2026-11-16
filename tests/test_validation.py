"""
Unit tests for data validation, file security, and schema compliance.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_validation import (
    validate_csv_file,
    validate_dataframe_schema,
    validate_inference_features,
)


def test_validate_csv_file_extensions(tmp_path):
    txt_file = tmp_path / "threat.txt"
    txt_file.write_text("dummy text")
    res = validate_csv_file(txt_file)
    assert res["is_valid"] is False
    assert "Invalid extension" in res["error"]

    csv_file = tmp_path / "traffic.csv"
    csv_file.write_text("a,b\n1,2")
    res_csv = validate_csv_file(csv_file)
    assert res_csv["is_valid"] is True


def test_validate_dataframe_schema():
    # Incomplete schema missing core features
    df_bad = pd.DataFrame({"dummy_col": [1, 2]})
    res_bad = validate_dataframe_schema(df_bad)
    assert res_bad["is_valid"] is False
    assert len(res_bad["missing_required_features"]) > 0

    # Valid schema
    valid_data = {
        "Destination Port": [80],
        "Flow Duration": [1000],
        "Total Fwd Packets": [5],
        "Total Backward Packets": [6],
        "Total Length of Fwd Packets": [500],
        "Total Length of Bwd Packets": [600],
        "Flow Bytes/s": [1000],
        "Flow Packets/s": [10],
        "Flow IAT Mean": [50],
        "Packet Length Mean": [100],
        "Label": ["BENIGN"],
    }
    df_good = pd.DataFrame(valid_data)
    res_good = validate_dataframe_schema(df_good)
    assert res_good["is_valid"] is True


def test_validate_inference_features():
    expected = ["feat1", "feat2", "feat3"]
    input_df = pd.DataFrame({"feat1": [1.0, 2.0], "extra_feat": [99, 99]})

    aligned, missing = validate_inference_features(input_df, expected)
    assert list(aligned.columns) == expected
    assert "feat2" in missing
    assert "feat3" in missing
    assert aligned["feat2"].iloc[0] == 0.0

