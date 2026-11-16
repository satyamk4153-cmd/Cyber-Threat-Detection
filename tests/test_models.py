"""
Unit tests for model instantiation, training, evaluation, and serialization.
"""

from pathlib import Path
import numpy as np
import pytest

from src.evaluate import evaluate_classifier, generate_model_comparison_table
from src.train import build_model, train_single_model


def test_build_model():
    dummy = build_model("dummy")
    assert dummy is not None

    lr = build_model("logistic_regression")
    assert lr is not None

    rf = build_model("random_forest")
    assert rf is not None

    with pytest.raises(ValueError):
        build_model("unknown_algorithm_xyz")


def test_train_and_evaluate_single_model():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.choice([0, 1], size=100)

    model, duration = train_single_model("decision_tree", X, y, random_state=42)
    assert duration >= 0.0

    metrics = evaluate_classifier(model, X, y, class_names=["BENIGN", "ATTACK"], training_time_sec=duration)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_generate_model_comparison_table():
    metrics_dict = {
        "model_a": {"accuracy": 0.90, "macro_f1": 0.88, "precision": 0.89, "recall": 0.87},
        "model_b": {"accuracy": 0.95, "macro_f1": 0.94, "precision": 0.93, "recall": 0.95},
    }
    table = generate_model_comparison_table(metrics_dict, selection_metric="macro_f1")
    assert len(table) == 2
    assert table.iloc[0]["model_name"] == "model_b"  # Higher macro_f1 first

