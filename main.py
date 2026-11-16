"""
Main execution entry point for Intelligent Cyber Threat Detection platform.
Provides a unified Command-Line Interface (CLI) to orchestrate data loading,
preprocessing, model training, evaluation, anomaly detection, and dashboard launching.
"""

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
from pathlib import Path
import argparse
import logging
import sys
import os

from src.config import config
from src.data_loader import check_raw_dataset_status, generate_sample_fixture, inspect_dataset, load_dataset
from src.preprocessing import prepare_train_test_split
from src.train import train_and_compare_suite
from src.evaluate import generate_model_comparison_table, save_evaluation_results
from src.anomaly_detection import NetworkAnomalyDetector
from src.explainability import extract_tree_feature_importance, calculate_permutation_importance
from src.predict import load_model_pipeline, predict_from_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CyberPulse")


def cmd_check_dataset():
    status = check_raw_dataset_status()
    print("\n" + "=" * 60)
    print("  CIC-IDS2017 DATASET ACQUISITION AUDIT")
    print("=" * 60)
    print(f"Target Directory: {status['directory']}")
    print(f"Raw Files Found:  {status['file_count']} files")
    print(f"Total Size:       {status['total_size_mb']:.2f} MB")
    print(f"Status:           {'OFFICIAL RAW FILES PRESENT' if status['is_available'] else 'MISSING RAW FILES (SOFTWARE FIXTURE MODE)'}")
    if not status["is_available"]:
        print("\nRequired Canonical CSV Files:")
        for rf in status["required_official_files"]:
            print(f"  - {rf}")
        print("\nDownload instructions: See docs/dataset_guide.md")
    print("=" * 60 + "\n")


def cmd_train_all(mode="binary"):
    print(f"\n🚀 Initiating End-to-End ML Training Pipeline (Mode: {mode.upper()})...")
    df, meta = load_dataset(use_sample_if_empty=True)
    print(f"Dataset Loaded: {len(df):,} flows from '{meta.get('source')}'.")

    models_to_run = [
        "dummy",
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    ]

    results = train_and_compare_suite(df, mode=mode, models_to_run=models_to_run)
    comp_table = generate_model_comparison_table(results["metrics"], selection_metric="macro_f1")
    csv_out, json_out = save_evaluation_results(comp_table, results["metrics"], mode=mode)

    print("\n" + "=" * 80)
    print(f"  MODEL BENCHMARK RESULTS ({mode.upper()})")
    print("=" * 80)
    cols = ["model_name", "accuracy", "precision", "recall", "macro_f1", "false_positive_rate", "training_time_sec"]
    print(comp_table[cols].to_string(index=False))
    print("=" * 80)
    print(f"Results exported to:\n  - {csv_out}\n  - {json_out}\n")


def cmd_anomaly():
    print("\n🔍 Initiating Unsupervised Anomaly Detection (Isolation Forest)...")
    df, _ = load_dataset(use_sample_if_empty=True)
    X_tr, X_te, y_tr, y_te, prep, _ = prepare_train_test_split(df, mode="binary")

    benign_mask = y_tr == 0
    detector = NetworkAnomalyDetector()
    detector.fit_benign_baseline(X_tr[benign_mask], preprocessor=prep)
    eval_res = detector.evaluate_against_ground_truth(X_te, y_te)
    saved_path = detector.save()

    print("\n" + "=" * 60)
    print("  ISOLATION FOREST ANOMALY EVALUATION")
    print("=" * 60)
    for k, v in eval_res.items():
        if k != "methodology_disclaimer":
            print(f"  {k:<35}: {v}")
    print("=" * 60)
    print(f"Model saved to: {saved_path}\n")


def cmd_explain():
    print("\n🧠 Extracting Model Feature Importance & Attribution...")
    try:
        pipeline = load_model_pipeline()
    except Exception as e:
        print(f"Error: {e}")
        return

    model = pipeline["model"]
    features = pipeline["features"]

    if hasattr(model, "feature_importances_"):
        imp_df = extract_tree_feature_importance(model, features)
        print("\n" + "=" * 60)
        print(f"  TOP 15 INFLUENTIAL FEATURES ({pipeline.get('model_name')})")
        print("=" * 60)
        for _, row in imp_df.head(15).iterrows():
            print(f"  {int(row['rank']):2d}. {row['feature']:<35} (Importance: {row['importance']:.4f})")
        print("=" * 60 + "\n")
    else:
        print("Active model does not use tree feature importances.")


def cmd_predict(csv_path: str):
    print(f"\n🧪 Running Offline Threat Inference on: {csv_path}...")
    try:
        results_df, summary = predict_from_csv(csv_path)
        print("\n" + "=" * 60)
        print("  INFERENCE RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Flows Analyzed: {summary['total_records_analyzed']:,}")
        print(f"Benign Traffic Flows: {summary['benign_records']:,}")
        print(f"Malicious Detections: {summary['attack_records']:,} ({summary['threat_ratio_percentage']:.1f}%)")
        print(f"Active Classifier:    {summary['model_used']}")
        print("\nRisk Tier Breakdown:")
        for tier, count in summary["risk_breakdown"].items():
            print(f"  - {tier:<10}: {count:,}")
        print("=" * 60 + "\n")

        out_path = Path("predictions_output.csv")
        results_df.to_csv(out_path, index=False)
        print(f"Detailed predictions saved to: {out_path.resolve()}\n")
    except Exception as e:
        print(f"Error running inference: {e}")


def cmd_launch_dashboard(port: int = 5000):
    """
    Launches web dashboard interface.
    Tries Streamlit if available; otherwise falls back seamlessly to the
    hardened Flask CyberPulse SOC portal.
    """
    import shutil
    import subprocess

    streamlit_available = shutil.which("streamlit") is not None
    if streamlit_available:
        try:
            import streamlit
            print("\n🛡️ Launching CyberPulse SOC Streamlit Dashboard...")
            app_file = Path("dashboard/app.py").resolve()
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_file), "--server.port", str(port)])
            return
        except ImportError:
            pass

    print(f"\n🛡️ Launching CyberPulse SOC Web Portal on http://127.0.0.1:{port}...")
    server_file = Path("dashboard/server.py").resolve()
    env = os.environ.copy()
    env["PORT"] = str(port)
    subprocess.run([sys.executable, str(server_file)], env=env)


def main():
    parser = argparse.ArgumentParser(
        description="CyberPulse: Intelligent Cyber Threat Detection & Network Traffic Analysis"
    )
    parser.add_argument("--check-dataset", action="store_true", help="Audit CIC-IDS2017 dataset acquisition status")
    parser.add_argument("--generate-sample", action="store_true", help="Generate software verification sample fixture")
    parser.add_argument("--train", action="store_true", help="Train all machine learning models and save artifacts")
    parser.add_argument("--multiclass", action="store_true", help="Run multiclass classification instead of binary")
    parser.add_argument("--anomaly", action="store_true", help="Train and evaluate Isolation Forest anomaly detection")
    parser.add_argument("--explain", action="store_true", help="Inspect model feature importance and explainability")
    parser.add_argument("--predict", type=str, metavar="CSV_PATH", help="Run batch threat prediction on input CSV file")
    parser.add_argument("--dashboard", action="store_true", help="Launch interactive SOC dashboard web portal")
    parser.add_argument("--server", action="store_true", help="Launch hardened Flask CyberPulse SOC web server")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")

    args = parser.parse_args()

    if args.check_dataset:
        cmd_check_dataset()
    elif args.generate_sample:
        out = generate_sample_fixture()
        print(f"Generated sample fixture at: {out}")
    elif args.train:
        mode = "multiclass" if args.multiclass else "binary"
        cmd_train_all(mode=mode)
    elif args.anomaly:
        cmd_anomaly()
    elif args.explain:
        cmd_explain()
    elif args.predict:
        cmd_predict(args.predict)
    elif args.dashboard or args.server:
        cmd_launch_dashboard(port=args.port)
    else:
        # Default: print status banner and run check
        print("\n" + "=" * 70)
        print("  🛡️  CYBERPULSE: INTELLIGENT CYBER THREAT DETECTION PLATFORM  🛡️")
        print("  Data Science & Defensive Network Traffic Analysis (CIC-IDS2017)")
        print("=" * 70)
        cmd_check_dataset()
        print("Available Commands:")
        print("  python main.py --train            (Train & evaluate ML models)")
        print("  python main.py --train --multiclass (Train multiclass attack models)")
        print("  python main.py --anomaly          (Run Isolation Forest anomaly detector)")
        print("  python main.py --explain          (Display influential feature rankings)")
        print("  python main.py --predict <file>   (Run offline threat inference on CSV)")
        print("  python main.py --dashboard        (Launch Streamlit SOC web dashboard)")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
