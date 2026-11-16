"""
Model explainability, feature importance, and diagnostic transparency module.
Implements Random Forest MDI importances, Permutation Importance on holdout data,
and optional SHAP explanations with defensive cybersecurity interpretations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.config import config

logger = logging.getLogger(__name__)


def extract_tree_feature_importance(
    model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Extracts Gini / Mean Decrease in Impurity (MDI) feature importances from tree-based estimators.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"Model of type {type(model)} does not provide feature_importances_ attribute.")

    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    df["cumulative_importance"] = df["importance"].cumsum()
    df["rank"] = range(1, len(df) + 1)
    return df


def calculate_permutation_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str = "f1_macro",
) -> pd.DataFrame:
    """
    Computes permutation feature importance on holdout test data.
    Permutation importance measures drop in performance when a feature is shuffled,
    providing a robust metric uncorrupted by feature cardinality or tree depth bias.
    """
    logger.info(f"Computing permutation importance with {n_repeats} repeats using scoring='{scoring}'...")
    perm_result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    df["rank"] = range(1, len(df) + 1)
    return df


def calculate_shap_values_sample(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    max_samples: int = 100,
) -> Optional[Dict[str, Any]]:
    """
    Computes SHAP (SHapley Additive exPlanations) values on a representative data sample
    if the shap package is available.
    """
    if not SHAP_AVAILABLE:
        logger.info("SHAP library is not installed or enabled. Skipping SHAP computation.")
        return None

    try:
        sample_subset = X_sample[:max_samples]
        # Use TreeExplainer for tree models, otherwise KernelExplainer/SamplingExplainer
        if hasattr(model, "estimators_") or hasattr(model, "tree_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_subset)
        else:
            explainer = shap.Explainer(model, sample_subset)
            shap_values = explainer(sample_subset).values

        # If multiclass or list of arrays, compute mean absolute SHAP per feature
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif shap_values.ndim == 3:
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        return {
            "is_available": True,
            "summary_df": shap_df.to_dict(orient="records"),
            "top_10": shap_df.head(10).to_dict(orient="records"),
        }
    except Exception as e:
        logger.warning(f"SHAP explanation calculation encountered an exception: {e}")
        return None


def get_causality_disclaimer() -> str:
    """Returns academic statement regarding feature importance vs causality."""
    return (
        "ACADEMIC RESEARCH NOTE: High feature importance indicates that an attribute is strongly "
        "predictive in the statistical decision boundaries of the trained model. "
        "It does NOT imply a direct physical or causal cybersecurity mechanism. In production networks, "
        "adversaries can dynamically manipulate flow statistics (e.g. packet padding, timing jitter), "
        "making continuous model governance essential."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_dataset
    from src.preprocessing import prepare_train_test_split
    from src.train import train_single_model

    df, _ = load_dataset(use_sample_if_empty=True)
    X_tr, X_te, y_tr, y_te, prep, meta = prepare_train_test_split(df, mode="binary")
    rf_model, _ = train_single_model("random_forest", X_tr, y_tr)

    imp_df = extract_tree_feature_importance(rf_model, prep.feature_names_)
    print("Top 5 Most Important Features (Gini MDI):")
    print(imp_df.head(5))

    perm_df = calculate_permutation_importance(rf_model, X_te, y_te, prep.feature_names_, n_repeats=3)
    print("\nTop 5 Permutation Importance Features:")
    print(perm_df.head(5))

