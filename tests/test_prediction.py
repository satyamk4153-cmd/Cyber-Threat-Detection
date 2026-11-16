"""
Unit tests for end-to-end inference, batch prediction lab, and anomaly detection.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import NetworkAnomalyDetector
from src.predict import load_model_pipeline, predict_traffic_records


def test_predict_traffic_records():
    # Load default trained pipeline
    pipeline = load_model_pipeline()
    assert "model" in pipeline
    assert "preprocessor" in pipeline

    # Create dummy records matching expected features
    features = pipeline["features"]
    dummy_data = {feat: np.random.uniform(10, 100, size=5) for feat in features}
    df_test = pd.DataFrame(dummy_data)

    results_df, summary = predict_traffic_records(df_test, pipeline)
    assert "Predicted_Threat_Label" in results_df.columns
    assert "Confidence_Score" in results_df.columns
    assert "Risk_Tier" in results_df.columns
    assert summary["total_records_analyzed"] == 5
    assert "threat_ratio_percentage" in summary


def test_anomaly_detection_pipeline():
    detector = NetworkAnomalyDetector()
    X_benign = np.random.randn(200, 15)
    detector.fit_benign_baseline(X_benign)

    X_test = np.random.randn(50, 15)
    preds, scores = detector.predict_anomalies(X_test)

    assert len(preds) == 50
    assert len(scores) == 50
    assert set(preds).issubset({0, 1})

