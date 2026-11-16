"""
Comprehensive model evaluation, diagnostic metrics, and comparative benchmarking module.
Computes Accuracy, Precision, Recall, Macro/Weighted F1, FPR, FNR, ROC-AUC, and latency.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import config

logger = logging.getLogger(__name__)


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    training_time_sec: float = 0.0,
    mode: str = "binary",
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on test data.
    Computes all standard and cybersecurity-specific metrics (FPR, FNR, latency).
    """
    # Measure prediction time
    start_infer = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start_infer

    # Latency per 1000 records
    latency_per_1k = round((inference_time / len(X_test)) * 1000, 5) if len(X_test) > 0 else 0.0

    # Probabilities if supported
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None
    elif hasattr(model, "decision_function"):
        try:
            df_vals = model.decision_function(X_test)
            # Min-max scale to [0, 1] as pseudo-probabilities
            if df_vals.ndim == 1:
                y_proba = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-9)
        except Exception:
            y_proba = None

    # Base Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_list = cm.tolist()

    # Cybersecurity-specific rates (False Positives & False Negatives)
    fpr: Optional[float] = None
    fnr: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    roc_data: Optional[Dict[str, Any]] = None

    if mode == "binary" and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        prec_bin = float(precision_score(y_test, y_pred, average="binary", zero_division=0))
        rec_bin = float(recall_score(y_test, y_pred, average="binary", zero_division=0))
        f1_bin = float(f1_score(y_test, y_pred, average="binary", zero_division=0))
    else:
        # Multiclass micro rates
        prec_bin = prec_macro
        rec_bin = rec_macro
        f1_bin = f1_macro

    # Compute ROC-AUC and PR-AUC when probabilities are available
    if y_proba is not None:
        try:
            if mode == "binary":
                prob_col = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                roc_auc = float(roc_auc_score(y_test, prob_col))
                pr_auc = float(average_precision_score(y_test, prob_col))
                fpr_curve, tpr_curve, thresholds = roc_curve(y_test, prob_col)
                # Store sample of curve points for UI/reports
                step = max(1, len(fpr_curve) // 100)
                roc_data = {
                    "fpr": [round(float(x), 4) for x in fpr_curve[::step]],
                    "tpr": [round(float(y), 4) for y in tpr_curve[::step]],
                    "auc": round(roc_auc, 4),
                }
            elif mode == "multiclass" and y_proba.ndim == 2 and y_proba.shape[1] == len(class_names):
                roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")

    # Per-class breakdown report
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names[:len(np.unique(y_test))],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec_bin, 4),
        "recall": round(rec_bin, 4),
        "f1_score": round(f1_bin, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "false_positive_rate": round(fpr, 4) if fpr is not None else None,
        "false_negative_rate": round(fnr, 4) if fnr is not None else None,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "training_time_sec": round(training_time_sec, 4),
        "inference_latency_ms_per_1k": round(latency_per_1k, 4),
        "confusion_matrix": cm_list,
        "class_names": class_names,
        "classification_report": report_dict,
        "roc_curve": roc_data,
    }

    return metrics


def generate_model_comparison_table(
    all_metrics: Dict[str, Dict[str, Any]],
    selection_metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Constructs a comparative benchmark summary DataFrame sorted by selection metric.
    """
    rows = []
    for model_name, m in all_metrics.items():
        rows.append({
            "model_name": model_name,
            "accuracy": m.get("accuracy"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1_score": m.get("f1_score"),
            "macro_f1": m.get("macro_f1"),
            "weighted_f1": m.get("weighted_f1"),
            "false_positive_rate": m.get("false_positive_rate"),
            "false_negative_rate": m.get("false_negative_rate"),
            "roc_auc": m.get("roc_auc"),
            "training_time_sec": m.get("training_time_sec"),
            "latency_ms_per_1k": m.get("inference_latency_ms_per_1k"),
        })

    df = pd.DataFrame(rows)
    if selection_metric in df.columns:
        df = df.sort_values(by=selection_metric, ascending=False).reset_index(drop=True)
    return df


def save_evaluation_results(
    comparison_df: pd.DataFrame,
    detailed_metrics: Dict[str, Any],
    mode: str = "binary",
) -> Tuple[Path, Path]:
    """Exports benchmark table and detailed metrics to reports/results/."""
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"model_comparison_{mode}.csv"
    json_path = results_dir / f"evaluation_metrics_{mode}.json"

    comparison_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, indent=2)

    logger.info(f"Saved comparison CSV to {csv_path} and JSON metrics to {json_path}")
    return csv_path, json_path

