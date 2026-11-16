"""
Feature engineering, domain-specific network traffic transformations,
and correlation/importance-based feature selection module.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from src.config import config

logger = logging.getLogger(__name__)


def compute_domain_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes defensive cybersecurity network traffic features derived from raw flow records.
    Only computes features for which base columns exist in the DataFrame.
    """
    featured_df = df.copy()
    cols = featured_df.columns

    # 1. Packet & Byte Ratios (Forward vs Backward)
    if "Total Fwd Packets" in cols and "Total Backward Packets" in cols:
        # Avoid division by zero with small epsilon
        featured_df["fwd_to_bwd_packet_ratio"] = (
            featured_df["Total Fwd Packets"] / (featured_df["Total Backward Packets"] + 1e-5)
        ).clip(0, 1000)

    if "Total Length of Fwd Packets" in cols and "Total Length of Bwd Packets" in cols:
        featured_df["fwd_to_bwd_byte_ratio"] = (
            featured_df["Total Length of Fwd Packets"] / (featured_df["Total Length of Bwd Packets"] + 1e-5)
        ).clip(0, 1000)

    # 2. Total Combined Flow Volumes
    if "Total Fwd Packets" in cols and "Total Backward Packets" in cols:
        featured_df["total_flow_packets"] = (
            featured_df["Total Fwd Packets"] + featured_df["Total Backward Packets"]
        )

    if "Total Length of Fwd Packets" in cols and "Total Length of Bwd Packets" in cols:
        featured_df["total_flow_bytes"] = (
            featured_df["Total Length of Fwd Packets"] + featured_df["Total Length of Bwd Packets"]
        )

    # 3. Log-transformed heavy-tailed metrics (Flow Duration, Bytes)
    if "Flow Duration" in cols:
        featured_df["log_flow_duration"] = np.log1p(
            featured_df["Flow Duration"].clip(lower=0)
        )

    if "Flow Bytes/s" in cols:
        featured_df["log_flow_bytes_per_s"] = np.log1p(
            featured_df["Flow Bytes/s"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        )

    # 4. TCP Flag Proportions / Combinations (Signature of PortScans & SYN Floods)
    flag_cols = [c for c in ["SYN Flag Count", "ACK Flag Count", "FIN Flag Count", "PSH Flag Count"] if c in cols]
    if len(flag_cols) >= 2:
        featured_df["total_tcp_control_flags"] = featured_df[flag_cols].sum(axis=1)

    if "SYN Flag Count" in cols and "ACK Flag Count" in cols:
        featured_df["syn_to_ack_ratio"] = (
            (featured_df["SYN Flag Count"] + 1e-5) / (featured_df["ACK Flag Count"] + 1e-5)
        ).clip(0, 100)

    # 5. Packet Size Coefficient of Variation (Mean vs Std)
    if "Packet Length Mean" in cols and "Packet Length Std" in cols:
        featured_df["packet_length_coeff_variation"] = (
            featured_df["Packet Length Std"] / (featured_df["Packet Length Mean"] + 1e-5)
        ).clip(0, 100)

    new_cols_added = [c for c in featured_df.columns if c not in cols]
    logger.info(f"Engineered {len(new_cols_added)} domain network features: {new_cols_added}")
    return featured_df


def identify_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str], List[Tuple[str, str, float]]]:
    """
    Computes correlation matrix and identifies collinear feature pairs exceeding threshold.
    Returns:
        (correlation_matrix, suggested_features_to_drop, correlated_pairs)
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    corr_matrix = numeric_df.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop: List[str] = []
    pairs: List[Tuple[str, str, float]] = []

    for col in upper_triangle.columns:
        high_corr = upper_triangle[col][upper_triangle[col] > threshold]
        for row_idx, val in high_corr.items():
            pairs.append((str(row_idx), str(col), round(float(val), 4)))
            if col not in to_drop:
                to_drop.append(col)

    logger.info(f"Identified {len(pairs)} feature pairs with correlation > {threshold}. Suggested drops: {len(to_drop)}")
    return corr_matrix, to_drop, pairs


def select_top_features_rf(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: int = 30,
    random_state: int = 42,
) -> Tuple[List[str], np.ndarray, Dict[str, float]]:
    """
    Uses a fast Random Forest feature selector to identify the top K most informative features.
    Guarantees that models are trained on the most predictive, non-redundant traffic signatures.
    """
    k = min(top_k, X.shape[1])
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:k]

    selected_features = [feature_names[i] for i in indices]
    importances_dict = {
        feature_names[i]: float(importances[i]) for i in indices
    }

    logger.info(f"Selected top {k} features via Random Forest importances.")
    return selected_features, indices, importances_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_dataset
    df, _ = load_dataset(use_sample_if_empty=True)
    df_feat = compute_domain_network_features(df)
    print("New shape after feature engineering:", df_feat.shape)
    _, drops, pairs = identify_highly_correlated_features(df_feat)
    print(f"Highly correlated pairs found: {len(pairs)}")

