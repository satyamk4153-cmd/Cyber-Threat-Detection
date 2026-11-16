"""
Unsupervised Anomaly Detection module using Isolation Forest.
Trained on baseline benign network flows to detect novel, out-of-distribution traffic patterns.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import config
from src.preprocessing import NetworkDataPreprocessor

logger = logging.getLogger(__name__)


class NetworkAnomalyDetector:
    """
    Unsupervised network traffic anomaly detector using Isolation Forest.
    Fits strictly on benign profile baseline traffic to identify behavioral deviations.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.15,
        max_samples: Union[str, int, float] = "auto",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.preprocessor: Optional[NetworkDataPreprocessor] = None
        self.threshold_: float = 0.0

    def fit_benign_baseline(
        self,
        X_benign: Union[pd.DataFrame, np.ndarray],
        preprocessor: Optional[NetworkDataPreprocessor] = None,
    ) -> "NetworkAnomalyDetector":
        """
        Fits the Isolation Forest strictly on known benign network traffic records.
        """
        self.preprocessor = preprocessor

        if isinstance(X_benign, pd.DataFrame):
            if self.preprocessor is not None:
                X_mat = self.preprocessor.transform(X_benign)
            else:
                X_mat = X_benign.values
        else:
            X_mat = X_benign

        logger.info(f"Fitting Isolation Forest on {len(X_mat)} benign baseline records...")
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_mat)

        # In IsolationForest, lower decision_function values imply higher anomaly
        scores = self.model.decision_function(X_mat)
        self.threshold_ = float(np.percentile(scores, self.contamination * 100))
        logger.info(f"Isolation Forest fitted. Empirical anomaly threshold: {self.threshold_:.4f}")
        return self

    def predict_anomalies(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts whether records are anomalous.
        Returns:
            (predictions: 1 for Anomaly, 0 for Normal;
             anomaly_scores: continuous severity score where lower means more anomalous)
        """
        if self.model is None:
            raise RuntimeError("Anomaly detector has not been fitted. Call fit_benign_baseline first.")

        if isinstance(X, pd.DataFrame):
            if self.preprocessor is not None:
                X_mat = self.preprocessor.transform(X)
            else:
                X_mat = X.values
        else:
            X_mat = X

        # raw_preds: -1 for anomaly, 1 for inlier
        raw_preds = self.model.predict(X_mat)
        # Convert to standard defensive flag: 1 = Anomaly (Threat Flag), 0 = Normal
        binary_anomaly = np.where(raw_preds == -1, 1, 0)

        # Decision function: negative values indicate outliers
        scores = self.model.decision_function(X_mat)
        return binary_anomaly, scores

    def evaluate_against_ground_truth(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compares unsupervised anomaly predictions with labeled test traffic (0: Benign, 1: Attack).
        Calculates Detection Rate (Recall on Attacks) and False Alarm Rate (FAR on Benign).
        """
        binary_preds, scores = self.predict_anomalies(X_test)

        # Ensure ground truth is binary: 0 = Benign, 1 = Attack
        y_true_binary = np.where(np.array(y_test) > 0, 1, 0)

        # Confusion elements
        tp = int(np.sum((binary_preds == 1) & (y_true_binary == 1)))  # Correctly flagged attacks
        fp = int(np.sum((binary_preds == 1) & (y_true_binary == 0)))  # Benign mistakenly flagged
        fn = int(np.sum((binary_preds == 0) & (y_true_binary == 1)))  # Attacks missed
        tn = int(np.sum((binary_preds == 0) & (y_true_binary == 0)))  # Benign correctly passed

        total_attacks = tp + fn
        total_benign = tn + fp

        detection_rate = (tp / total_attacks) if total_attacks > 0 else 0.0
        false_alarm_rate = (fp / total_benign) if total_benign > 0 else 0.0
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1_anomaly = (
            2 * (precision * detection_rate) / (precision + detection_rate)
            if (precision + detection_rate) > 0
            else 0.0
        )

        results = {
            "total_evaluated": len(binary_preds),
            "anomalies_detected": int(np.sum(binary_preds == 1)),
            "normal_detected": int(np.sum(binary_preds == 0)),
            "anomaly_percentage": round(float(np.mean(binary_preds == 1) * 100), 2),
            "true_positives_attacks_flagged": tp,
            "false_positives_benign_alarms": fp,
            "false_negatives_attacks_missed": fn,
            "true_negatives_benign_passed": tn,
            "detection_rate_recall": round(float(detection_rate), 4),
            "false_alarm_rate_fpr": round(float(false_alarm_rate), 4),
            "anomaly_precision": round(float(precision), 4),
            "f1_score": round(float(f1_anomaly), 4),
            "mean_anomaly_score": round(float(np.mean(scores)), 4),
            "min_anomaly_score": round(float(np.min(scores)), 4),
            "max_anomaly_score": round(float(np.max(scores)), 4),
            "decision_threshold": round(self.threshold_, 4),
            "methodology_disclaimer": (
                "Unsupervised anomaly detection flags statistical deviations from baseline normal traffic. "
                "An anomaly indicates unusual behavior or novel zero-day patterns, but is NOT a definitive, "
                "forensically confirmed cyberattack signature."
            ),
        }
        return results

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Saves fitted anomaly detector to disk."""
        target_path = Path(filepath) if filepath else config.saved_models_dir / "isolation_forest.joblib"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)
        logger.info(f"Saved anomaly detector to {target_path}")
        return target_path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "NetworkAnomalyDetector":
        target_path = Path(filepath) if filepath else config.saved_models_dir / "isolation_forest.joblib"
        if not target_path.exists():
            raise FileNotFoundError(f"Anomaly detector model not found at {target_path}")
        return joblib.load(target_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_dataset
    from src.preprocessing import prepare_train_test_split

    df, _ = load_dataset(use_sample_if_empty=True)
    X_tr, X_te, y_tr, y_te, prep, _ = prepare_train_test_split(df, mode="binary")

    # Fit detector on benign records from training set
    benign_mask = y_tr == 0
    detector = NetworkAnomalyDetector()
    detector.fit_benign_baseline(X_tr[benign_mask])

    res = detector.evaluate_against_ground_truth(X_te, y_te)
    print("Anomaly Detection Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")

