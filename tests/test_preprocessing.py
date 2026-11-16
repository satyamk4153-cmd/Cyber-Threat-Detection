"""
Unit tests for data cleaning, preprocessing, and train/test leakage prevention.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import NetworkDataPreprocessor, clean_labels, prepare_train_test_split


def test_clean_labels():
    s = pd.Series(["  BENIGN ", "Web Attack  Brute Force", "DoS Hulk "])
    cleaned = clean_labels(s)
    assert cleaned.iloc[0] == "BENIGN"
    assert cleaned.iloc[1] == "Web Attack - Brute Force"
    assert cleaned.iloc[2] == "DoS Hulk"


def test_preprocessor_pipeline():
    df_train = pd.DataFrame({
        "Flow Duration": [100.0, 200.0, np.nan, 400.0],
        "Flow Bytes/s": [10.0, np.inf, 30.0, 40.0],
        "Constant_Col": [1.0, 1.0, 1.0, 1.0],  # Should be dropped
        "Label": ["BENIGN", "DoS Hulk", "BENIGN", "DDoS"],
    })

    preprocessor = NetworkDataPreprocessor()
    X = df_train.drop(columns=["Label"])
    y = df_train["Label"]

    X_scaled = preprocessor.fit_transform(X, y)
    assert "Constant_Col" in preprocessor.dropped_constant_features_
    assert "Constant_Col" not in preprocessor.feature_names_
    assert not np.isnan(X_scaled).any()
    assert not np.isinf(X_scaled).any()


def test_prepare_train_test_split():
    df = pd.DataFrame({
        "Destination Port": [80, 443, 22, 8080, 53, 445, 80, 443],
        "Flow Duration": [100, 200, 300, 400, 500, 600, 700, 800],
        "Label": ["BENIGN", "BENIGN", "BENIGN", "BENIGN", "DoS Hulk", "DoS Hulk", "DDoS", "DDoS"],
    })

    X_tr, X_te, y_tr, y_te, prep, meta = prepare_train_test_split(df, mode="binary", test_size=0.25, random_state=42)
    assert len(X_tr) == 6
    assert len(X_te) == 2
    assert meta["mode"] == "binary"
    assert len(meta["classes"]) == 2

