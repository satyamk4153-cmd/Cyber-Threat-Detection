"""
Data cleaning and preprocessing pipeline for network intrusion traffic.
Prevents train/test data leakage, handles missing/infinite values, and standardizes features.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config import config, PROJECT_ROOT

logger = logging.getLogger(__name__)


def clean_labels(series: pd.Series) -> pd.Series:
    """Normalizes label strings by stripping whitespace and standardizing known variants."""
    s = series.astype(str).str.strip()
    # Normalize common encoding glitches in CIC-IDS2017
    s = s.replace({
        "Web Attack  Brute Force": "Web Attack - Brute Force",
        "Web Attack  XSS": "Web Attack - XSS",
        "Web Attack  Sql Injection": "Web Attack - Sql Injection",
    })
    return s


class NetworkDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Stateful preprocessor for network intrusion detection records.
    Learns medians, scaling parameters, outlier bounds, and active feature sets
    strictly on training data to guarantee zero data leakage.
    """

    def __init__(
        self,
        imputation_strategy: str = "median",
        scaler_type: str = "StandardScaler",
        outlier_clipping: bool = True,
        lower_quantile: float = 0.001,
        upper_quantile: float = 0.999,
        drop_constant: bool = True,
        variance_threshold: float = 0.0,
        leakage_features: Optional[List[str]] = None,
    ):
        self.imputation_strategy = imputation_strategy
        self.scaler_type = scaler_type
        self.outlier_clipping = outlier_clipping
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.drop_constant = drop_constant
        self.variance_threshold = variance_threshold
        self.leakage_features = leakage_features or config.leakage_features

        # Learned attributes
        self.feature_names_: List[str] = []
        self.dropped_constant_features_: List[str] = []
        self.dropped_leakage_features_: List[str] = []
        self.clip_lower_bounds_: Dict[str, float] = {}
        self.clip_upper_bounds_: Dict[str, float] = {}
        self.imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.class_names_: List[str] = []
        self.binary_mapping_: Dict[str, int] = {config.benign_label: 0, config.attack_label: 1}

    def _sanitize_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sanitizes column names, converts numbers, and converts infinities to NaN."""
        df = X.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # Convert infinities to NaN for imputation
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop known identifier/leakage columns
        leakage_cols = [c for c in self.leakage_features if c in df.columns]
        if leakage_cols:
            df = df.drop(columns=leakage_cols)

        # Force numeric conversion on remaining feature columns
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "NetworkDataPreprocessor":
        """
        Fits the imputation, clipping, and scaling transformations on training records only.
        """
        df = self._sanitize_dataframe(X)

        # Track and remove constant columns (variance <= threshold)
        if self.drop_constant:
            variances = df.var(skipna=True)
            self.dropped_constant_features_ = list(
                variances[variances <= self.variance_threshold].index
            )
            df = df.drop(columns=self.dropped_constant_features_)
            logger.info(f"Dropped {len(self.dropped_constant_features_)} constant features during fit.")

        self.feature_names_ = list(df.columns)

        # Compute outlier quantile boundaries on training data
        if self.outlier_clipping:
            for col in self.feature_names_:
                q_low = float(df[col].quantile(self.lower_quantile))
                q_high = float(df[col].quantile(self.upper_quantile))
                self.clip_lower_bounds_[col] = q_low
                self.clip_upper_bounds_[col] = q_high
                # Apply clipping to fit imputer/scaler robustly
                df[col] = df[col].clip(q_low, q_high)

        # Fit imputer
        self.imputer_ = SimpleImputer(strategy=self.imputation_strategy)
        imputed_array = self.imputer_.fit_transform(df)

        # Fit scaler
        self.scaler_ = StandardScaler()
        self.scaler_.fit(imputed_array)

        # Fit label encoder if multiclass y provided
        if y is not None:
            cleaned_y = clean_labels(y)
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(cleaned_y)
            self.class_names_ = list(self.label_encoder_.classes_)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Applies learned transformations to test/inference records without re-learning parameters.
        """
        if self.imputer_ is None or self.scaler_ is None:
            raise RuntimeError("Preprocessor has not been fitted. Call 'fit' before 'transform'.")

        df = self._sanitize_dataframe(X)

        # Ensure all fitted features are present; add missing as 0.0
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0.0

        # Restrict strictly to fitted features in fitted order
        df = df[self.feature_names_].copy()

        # Apply clipping using training-learned boundaries
        if self.outlier_clipping:
            for col in self.feature_names_:
                low = self.clip_lower_bounds_.get(col, -np.inf)
                high = self.clip_upper_bounds_.get(col, np.inf)
                df[col] = df[col].clip(low, high)

        imputed_array = self.imputer_.transform(df)
        scaled_array = self.scaler_.transform(imputed_array)
        return scaled_array

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def encode_labels(
        self, y: pd.Series, mode: str = "binary"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Encodes target labels into numerical values.
        Supports binary ('BENIGN' vs 'ATTACK') and multiclass.
        """
        cleaned_y = clean_labels(y)

        if mode == "binary":
            # 0 for BENIGN, 1 for any attack
            benign_val = config.benign_label.upper()
            binary_arr = np.where(cleaned_y.str.upper() == benign_val, 0, 1)
            class_labels = [config.benign_label, config.attack_label]
            weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=binary_arr)
            class_weights_dict = {0: float(weights[0]), 1: float(weights[1])}
            return binary_arr, {
                "mode": "binary",
                "classes": class_labels,
                "class_weights": class_weights_dict,
                "mapping": {config.benign_label: 0, config.attack_label: 1},
            }

        elif mode == "multiclass":
            if self.label_encoder_ is None:
                self.label_encoder_ = LabelEncoder()
                encoded_arr = self.label_encoder_.fit_transform(cleaned_y)
                self.class_names_ = list(self.label_encoder_.classes_)
            else:
                encoded_arr = self.label_encoder_.transform(cleaned_y)

            classes_arr = np.unique(encoded_arr)
            weights = compute_class_weight("balanced", classes=classes_arr, y=encoded_arr)
            class_weights_dict = {int(c): float(w) for c, w in zip(classes_arr, weights)}

            mapping = {name: int(idx) for idx, name in enumerate(self.label_encoder_.classes_)}
            return encoded_arr, {
                "mode": "multiclass",
                "classes": list(self.label_encoder_.classes_),
                "class_weights": class_weights_dict,
                "mapping": mapping,
            }

        else:
            raise ValueError(f"Unsupported encoding mode '{mode}'. Choose 'binary' or 'multiclass'.")

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes the fitted preprocessor pipeline to disk."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info(f"Saved preprocessor to {p}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "NetworkDataPreprocessor":
        """Loads a saved preprocessor pipeline from disk."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Preprocessor file not found at {p}")
        return joblib.load(p)


def prepare_train_test_split(
    df: pd.DataFrame,
    mode: str = "binary",
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, NetworkDataPreprocessor, Dict[str, Any]]:
    """
    High-level helper to perform stratified train/test split and fit preprocessor on training data only.
    Guarantees no data leakage.

    Returns:
        (X_train_scaled, X_test_scaled, y_train, y_test, fitted_preprocessor, target_meta)
    """
    target_col = config.target_column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Drop duplicate records prior to splitting
    initial_len = len(df)
    clean_df = df.drop_duplicates().reset_index(drop=True)
    dropped_dupes = initial_len - len(clean_df)
    if dropped_dupes > 0:
        logger.info(f"Removed {dropped_dupes} duplicate rows prior to splitting.")

    X = clean_df.drop(columns=[target_col])
    y_raw = clean_df[target_col]

    preprocessor = NetworkDataPreprocessor()

    # Encode target labels
    y_encoded, target_meta = preprocessor.encode_labels(y_raw, mode=mode)

    # Stratified split to preserve class imbalance ratios
    stratify_target = y_encoded if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    logger.info(f"Train split: {len(X_train)} samples, Test split: {len(X_test)} samples.")

    # Fit preprocessor strictly on X_train
    X_train_scaled = preprocessor.fit_transform(X_train, y=y_raw.iloc[X_train.index])
    # Transform X_test using learned parameters
    X_test_scaled = preprocessor.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor, target_meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_dataset

    df, _ = load_dataset(use_sample_if_empty=True)
    X_tr, X_te, y_tr, y_te, prep, meta = prepare_train_test_split(df, mode="binary")
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
    print("Class metadata:", meta)
    print("Features selected:", len(prep.feature_names_))

