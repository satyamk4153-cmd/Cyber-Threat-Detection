"""
Machine learning training pipeline for network intrusion detection.
Supports Logistic Regression, Decision Tree, Random Forest, Gradient Boosting,
XGBoost, and Baseline Dummy Classifier across Binary and Multiclass tasks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.config import config, PROJECT_ROOT
from src.preprocessing import NetworkDataPreprocessor, prepare_train_test_split

logger = logging.getLogger(__name__)


def build_model(
    model_type: str,
    class_weights: Optional[Dict[int, float]] = None,
    random_state: int = 42,
) -> Any:
    """
    Instantiates an algorithm with calibrated hyperparameters from config.yaml.
    """
    m_type = model_type.lower().strip()

    if m_type == "dummy":
        return DummyClassifier(strategy="stratified", random_state=random_state)

    elif m_type in ["logistic_regression", "lr"]:
        cfg = config.get("models.logistic_regression", {})
        return LogisticRegression(
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight=class_weights or cfg.get("class_weight", "balanced"),
            random_state=random_state,
        )

    elif m_type in ["decision_tree", "dt"]:
        cfg = config.get("models.decision_tree", {})
        return DecisionTreeClassifier(
            criterion=cfg.get("criterion", "gini"),
            max_depth=cfg.get("max_depth", 15),
            min_samples_split=cfg.get("min_samples_split", 10),
            min_samples_leaf=cfg.get("min_samples_leaf", 5),
            class_weight=class_weights or cfg.get("class_weight", "balanced"),
            random_state=random_state,
        )

    elif m_type in ["random_forest", "rf"]:
        cfg = config.get("models.random_forest", {})
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", 20),
            min_samples_split=cfg.get("min_samples_split", 8),
            min_samples_leaf=cfg.get("min_samples_leaf", 4),
            class_weight=class_weights or cfg.get("class_weight", "balanced"),
            n_jobs=cfg.get("n_jobs", -1),
            random_state=random_state,
        )

    elif m_type in ["gradient_boosting", "gb"]:
        cfg = config.get("models.gradient_boosting", {})
        return GradientBoostingClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            learning_rate=cfg.get("learning_rate", 0.1),
            max_depth=cfg.get("max_depth", 6),
            subsample=cfg.get("subsample", 0.8),
            random_state=random_state,
        )

    elif m_type in ["xgboost", "xgb"]:
        if not XGB_AVAILABLE:
            logger.warning("XGBoost library not found. Falling back to GradientBoostingClassifier.")
            return build_model("gradient_boosting", class_weights, random_state)

        cfg = config.get("models.xgboost", {})
        return xgb.XGBClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            learning_rate=cfg.get("learning_rate", 0.1),
            max_depth=cfg.get("max_depth", 6),
            eval_metric=cfg.get("eval_metric", "logloss"),
            random_state=random_state,
            n_jobs=-1,
        )

    else:
        raise ValueError(
            f"Unsupported model type '{model_type}'. Choose from: 'dummy', 'logistic_regression', "
            "'decision_tree', 'random_forest', 'gradient_boosting', 'xgboost'."
        )


def train_single_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: Optional[Dict[int, float]] = None,
    random_state: int = 42,
) -> Tuple[Any, float]:
    """Trains a single model instance and records duration."""
    logger.info(f"Initiating training for: {model_type}...")
    model = build_model(model_type, class_weights=class_weights, random_state=random_state)

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = round(time.perf_counter() - start_time, 4)

    logger.info(f"Finished training {model_type} in {training_time:.3f} seconds.")
    return model, training_time


def save_trained_pipeline(
    model: Any,
    preprocessor: NetworkDataPreprocessor,
    model_name: str,
    target_mode: str,
    metrics: Dict[str, Any],
    save_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """
    Saves complete deployment-ready artifact containing:
    1. Preprocessor + trained model (joblib)
    2. Comprehensive metadata JSON (features, hyperparameters, metrics, timestamps)
    """
    models_dir = Path(save_dir) if save_dir else config.saved_models_dir
    meta_dir = config.model_metadata_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    filename_base = f"{model_name.lower()}_{target_mode}"
    model_file = models_dir / f"{filename_base}.joblib"
    meta_file = meta_dir / f"{filename_base}_metadata.json"

    pipeline_package = {
        "model": model,
        "preprocessor": preprocessor,
        "model_name": model_name,
        "target_mode": target_mode,
        "features": preprocessor.feature_names_,
        "classes": getattr(preprocessor, "class_names_", ["BENIGN", "ATTACK"]),
        "trained_at": datetime.now().isoformat(),
        "sklearn_version": joblib.__version__,
    }

    joblib.dump(pipeline_package, model_file)

    metadata = {
        "model_name": model_name,
        "target_mode": target_mode,
        "model_type": model.__class__.__name__,
        "n_features": len(preprocessor.feature_names_),
        "feature_names": preprocessor.feature_names_,
        "dropped_constant_features": preprocessor.dropped_constant_features_,
        "classes": preprocessor.class_names_ if hasattr(preprocessor, "class_names_") else ["BENIGN", "ATTACK"],
        "metrics": metrics,
        "model_file": str(model_file.name),
        "created_at": datetime.now().isoformat(),
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved model pipeline to {model_file} and metadata to {meta_file}")
    return model_file, meta_file


def train_and_compare_suite(
    df: pd.DataFrame,
    mode: str = "binary",
    models_to_run: Optional[List[str]] = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Runs full training and evaluation suite across configured algorithms.
    Returns dictionary with trained models, preprocessor, and performance metrics.
    """
    from src.evaluate import evaluate_classifier

    suite_models = models_to_run or [
        "dummy",
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    ]

    logger.info(f"Starting training suite for {mode} classification with {len(suite_models)} models.")
    X_train, X_test, y_train, y_test, preprocessor, target_meta = prepare_train_test_split(
        df, mode=mode, test_size=test_size, random_state=random_state
    )

    results: Dict[str, Any] = {
        "target_mode": mode,
        "target_meta": target_meta,
        "models": {},
        "metrics": {},
        "preprocessor": preprocessor,
        "test_data": (X_test, y_test),
    }

    for m_type in suite_models:
        try:
            model, train_time = train_single_model(
                m_type,
                X_train,
                y_train,
                class_weights=target_meta.get("class_weights"),
                random_state=random_state,
            )
            # Evaluate model
            m_metrics = evaluate_classifier(
                model=model,
                X_test=X_test,
                y_test=y_test,
                class_names=target_meta.get("classes", ["BENIGN", "ATTACK"]),
                training_time_sec=train_time,
                mode=mode,
            )
            results["models"][m_type] = model
            results["metrics"][m_type] = m_metrics

            # Save model to disk
            save_trained_pipeline(
                model=model,
                preprocessor=preprocessor,
                model_name=m_type,
                target_mode=mode,
                metrics=m_metrics,
            )
        except Exception as e:
            logger.error(f"Error training model '{m_type}': {e}", exc_info=True)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_dataset
    df, _ = load_dataset(use_sample_if_empty=True)
    res = train_and_compare_suite(df, mode="binary", models_to_run=["dummy", "decision_tree", "random_forest"])
    print("Models trained successfully:", list(res["models"].keys()))

