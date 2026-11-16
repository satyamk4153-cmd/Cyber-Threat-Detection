"""
Inference and prediction module for offline network traffic evaluation and CSV lab uploads.
Performs schema validation, feature alignment, and generates confidence-scored threat classifications.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import joblib
import numpy as np
import pandas as pd

from src.config import config
from src.data_validation import validate_csv_file, validate_dataframe_schema, validate_inference_features

logger = logging.getLogger(__name__)


def load_model_pipeline(model_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Loads saved model pipeline package containing:
    model, preprocessor, feature_names, class_names, target_mode, etc.
    """
    if model_path:
        p = Path(model_path)
    else:
        # Default to best performing model or random forest binary
        default_p = config.saved_models_dir / "random_forest_binary.joblib"
        if default_p.exists():
            p = default_p
        else:
            # Fall back to any joblib in saved_models
            available = list(config.saved_models_dir.glob("*.joblib"))
            if available:
                p = available[0]
            else:
                raise FileNotFoundError(
                    f"No trained models found in {config.saved_models_dir}. Please run model training first."
                )

    if not p.exists():
        raise FileNotFoundError(f"Model pipeline artifact does not exist at: {p}")

    package = joblib.load(p)
    logger.info(f"Loaded model pipeline package: {package.get('model_name')} ({package.get('target_mode')})")
    return package


def predict_traffic_records(
    df: pd.DataFrame,
    pipeline_package: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs end-to-end inference on an input DataFrame using a loaded model package.
    Returns:
        (annotated_df_with_predictions, summary_stats_dict)
    """
    model = pipeline_package["model"]
    preprocessor = pipeline_package["preprocessor"]
    expected_features = pipeline_package["features"]
    classes = pipeline_package["classes"]
    target_mode = pipeline_package.get("target_mode", "binary")

    # Step 1: Align input columns to match expected feature space
    aligned_df, missing_features = validate_inference_features(df, expected_features)

    # Step 2: Transform features using the frozen preprocessor pipeline
    X_scaled = preprocessor.transform(aligned_df)

    # Step 3: Run model prediction
    raw_preds = model.predict(X_scaled)

    # Step 4: Compute confidence / class probabilities if supported
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_scaled)
        except Exception as e:
            logger.debug(f"predict_proba unavailable: {e}")

    # Step 5: Map predictions to readable labels and risk tiers
    results_df = df.copy()
    if target_mode == "binary":
        label_names = [config.attack_label if p == 1 else config.benign_label for p in raw_preds]
        if probabilities is not None and probabilities.shape[1] >= 2:
            conf_scores = [round(float(probabilities[i, int(raw_preds[i])]), 4) for i in range(len(raw_preds))]
        else:
            conf_scores = [1.0] * len(raw_preds)
    else:
        # Multiclass: raw_preds are integer class indices
        label_names = [
            classes[p] if 0 <= p < len(classes) else f"Unknown_Class_{p}"
            for p in raw_preds
        ]
        if probabilities is not None and probabilities.shape[1] == len(classes):
            conf_scores = [round(float(probabilities[i, int(raw_preds[i])]), 4) for i in range(len(raw_preds))]
        else:
            conf_scores = [1.0] * len(raw_preds)

    # Calculate Risk Tiers
    risk_tiers = []
    for lbl, conf in zip(label_names, conf_scores):
        if lbl.upper() == config.benign_label.upper():
            risk_tiers.append("LOW" if conf > 0.7 else "INFO")
        elif "DDOS" in lbl.upper() or "HULK" in lbl.upper() or "INFILTRATION" in lbl.upper():
            risk_tiers.append("CRITICAL" if conf > 0.8 else "HIGH")
        else:
            risk_tiers.append("MEDIUM" if conf > 0.6 else "WARNING")

    results_df["Predicted_Threat_Label"] = label_names
    results_df["Confidence_Score"] = conf_scores
    results_df["Risk_Tier"] = risk_tiers

    total_records = len(results_df)
    attack_records = int(sum(1 for lbl in label_names if lbl.upper() != config.benign_label.upper()))
    benign_records = total_records - attack_records

    summary = {
        "model_used": pipeline_package.get("model_name"),
        "target_mode": target_mode,
        "total_records_analyzed": total_records,
        "benign_records": benign_records,
        "attack_records": attack_records,
        "threat_ratio_percentage": round((attack_records / total_records * 100), 2) if total_records > 0 else 0.0,
        "missing_features_imputed": len(missing_features),
        "risk_breakdown": pd.Series(risk_tiers).value_counts().to_dict(),
        "threat_breakdown": pd.Series(label_names).value_counts().to_dict(),
    }

    return results_df, summary


def predict_from_csv(
    csv_file_or_path: Union[str, Path, Any],
    model_pipeline_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    High-level entrypoint for CSV prediction laboratory.
    Validates CSV file size, reads safely, validates schema, and runs inference.
    """
    # Validate file integrity and size
    val_file = validate_csv_file(csv_file_or_path)
    if not val_file["is_valid"]:
        raise ValueError(f"CSV Validation Error: {val_file.get('error')}")

    # Read CSV
    df = pd.read_csv(csv_file_or_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # Validate schema
    schema_status = validate_dataframe_schema(df, allow_missing_target=True)
    if not schema_status["is_valid"]:
        raise ValueError(f"Schema Validation Error: {schema_status.get('error')}")

    # Load model and predict
    package = load_model_pipeline(model_pipeline_path)
    return predict_traffic_records(df, package)

