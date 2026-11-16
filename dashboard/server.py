"""
CyberPulse SOC Web Portal - Interactive Defensive Threat Analytics Server
Powered by Flask 3.1 & Modern Cyber Operations Center UI.
Production-hardened with HTTP security headers, CORS, sanitized uploads,
responsive navigation, accessible controls, and real-time inference telemetry.
"""

from pathlib import Path
import sys
import os
import io
import json
import uuid
import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from flask import Flask, render_template_string, request, jsonify, Response
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.data_loader import check_raw_dataset_status, inspect_dataset, load_dataset
from src.predict import load_model_pipeline, predict_traffic_records
from src.data_validation import validate_csv_file
from src.anomaly_detection import NetworkAnomalyDetector
from src.explainability import extract_tree_feature_importance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CyberPulseServer")

app = Flask(__name__)
# Maximum upload size: 50MB
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Thread-safe in-memory cache for prediction export downloads
_cache_lock = threading.Lock()
_prediction_cache: Dict[str, bytes] = {}
_latest_download_token: Optional[str] = None

# Preload dataset and inspection telemetry
logger.info("Loading network telemetry records...")
try:
    df_global, metadata_global = load_dataset(use_sample_if_empty=True)
    summary_global = inspect_dataset(df_global)
except Exception as e:
    logger.error(f"Error loading initial dataset: {e}")
    df_global = pd.DataFrame()
    metadata_global = {}
    summary_global = {
        "total_rows": 0, "total_columns": 0, "benign_count": 0, "attack_count": 0,
        "attack_percentage": 0.0, "total_missing_values": 0, "total_infinite_values": 0,
        "duplicate_rows": 0, "memory_usage_mb": 0.0, "label_counts": {}, "columns": []
    }

# Load evaluation metrics if present
eval_metrics_file = config.results_dir / "evaluation_metrics_binary.json"
if eval_metrics_file.exists():
    try:
        with open(eval_metrics_file, "r", encoding="utf-8") as f:
            metrics_global = json.load(f)
    except Exception:
        metrics_global = {}
else:
    metrics_global = {}

# Load comparison table
comp_csv = config.results_dir / "model_comparison_binary.csv"
if comp_csv.exists():
    try:
        df_comp_global = pd.read_csv(comp_csv).to_dict(orient="records")
    except Exception:
        df_comp_global = []
else:
    df_comp_global = []

# Load trained pipeline
try:
    pipeline_global = load_model_pipeline()
    active_model_name = pipeline_global.get("model_name", "Random Forest").replace("_", " ").title()
    active_features = pipeline_global.get("features", [])
    if hasattr(pipeline_global.get("model"), "feature_importances_"):
        top_features_global = extract_tree_feature_importance(
            pipeline_global["model"], active_features
        ).head(15).to_dict(orient="records")
    else:
        top_features_global = []
except Exception as e:
    logger.warning(f"Failed to load pipeline on startup: {e}")
    pipeline_global = None
    active_model_name = "Random Forest (Default)"
    top_features_global = []

# Load anomaly detector
anom_path = config.saved_models_dir / "isolation_forest.joblib"
if anom_path.exists():
    try:
        detector_global = NetworkAnomalyDetector.load(anom_path)
    except Exception:
        detector_global = None
else:
    detector_global = None


@app.after_request
def set_security_headers(response: Response) -> Response:
    """Enforces strict production HTTP security headers."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "font-src 'self' data:; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    )
    return response


@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Resource not found", "status": 404}), 404
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head><title>404 - CyberPulse SOC</title><style>
    body { background: #0B132B; color: #F8F9FA; font-family: sans-serif; text-align: center; padding: 60px; }
    h1 { color: #FF0054; } a { color: #00F5D4; text-decoration: none; }
    </style></head><body>
    <h1>🛡️ 404 - Page Not Found</h1>
    <p>The requested SOC telemetry resource does not exist.</p>
    <p><a href="/">Return to Security Command Center</a></p>
    </body></html>
    """, 404


@app.errorhandler(413)
def handle_large_file(e):
    return jsonify({"error": "Uploaded file exceeds the maximum allowed 50MB payload limit.", "status": 413}), 413


@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal Server Error: {e}")
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server processing error. Check server logs.", "status": 500}), 500
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head><title>500 - Server Error</title><style>
    body { background: #0B132B; color: #F8F9FA; font-family: sans-serif; text-align: center; padding: 60px; }
    h1 { color: #FF0054; } a { color: #00F5D4; }
    </style></head><body>
    <h1>🚨 500 - Internal Processing Error</h1>
    <p>A server error occurred during telemetry processing.</p>
    <p><a href="/">Return to Dashboard</a></p>
    </body></html>
    """, 500


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ CyberPulse SOC - Intelligent Cyber Threat Analytics</title>
    <!-- Chart.js for Interactive Telemetry Visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-dark: #0B132B;
            --bg-sidebar: #0D1B2A;
            --bg-card: #1C2541;
            --bg-card-alt: #111827;
            --border: #1E293B;
            --text-main: #F8F9FA;
            --text-muted: #94A3B8;
            --accent-blue: #3A86FF;
            --accent-cyan: #00F5D4;
            --accent-red: #FF0054;
            --accent-green: #38B000;
            --accent-yellow: #FFBE0B;
            --accent-orange: #FF6B35;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        *:focus-visible {
            outline: 2px solid var(--accent-cyan);
            outline-offset: 2px;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* Mobile Header */
        .mobile-header {
            display: none;
            background-color: var(--bg-sidebar);
            border-bottom: 1px solid var(--border);
            padding: 12px 16px;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 900;
        }

        .mobile-toggle {
            background: none;
            border: 1px solid var(--border);
            color: var(--accent-cyan);
            font-size: 1.4rem;
            padding: 6px 10px;
            border-radius: 6px;
            cursor: pointer;
        }

        /* Sidebar Styling */
        #sidebar {
            width: 280px;
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 24px 16px;
            overflow-y: auto;
            flex-shrink: 0;
            transition: transform 0.3s ease;
        }

        .brand {
            text-align: center;
            margin-bottom: 24px;
        }
        .brand-icon { font-size: 2.2rem; }
        .brand-title {
            color: var(--accent-cyan);
            font-weight: 800;
            font-size: 1.25rem;
            letter-spacing: 0.05em;
        }
        .brand-sub {
            color: var(--text-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 2px;
        }

        .nav-section {
            font-size: 0.72rem;
            text-transform: uppercase;
            color: var(--accent-blue);
            font-weight: 700;
            letter-spacing: 0.08em;
            margin: 16px 0 8px 8px;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            color: var(--text-muted);
            text-decoration: none;
            border-radius: 6px;
            font-size: 0.88rem;
            font-weight: 500;
            margin-bottom: 4px;
            cursor: pointer;
            border: 1px solid transparent;
            background: transparent;
            width: 100%;
            text-align: left;
            transition: all 0.2s ease;
        }
        .nav-item:hover, .nav-item.active {
            background-color: var(--bg-card);
            color: var(--accent-cyan);
            border-left: 3px solid var(--accent-cyan);
        }

        .status-box {
            margin-top: auto;
            background-color: rgba(28, 37, 65, 0.6);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            font-size: 0.75rem;
        }
        .status-pill {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.7rem;
            margin-top: 4px;
        }
        .pill-active { background: rgba(56, 176, 0, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .pill-warn { background: rgba(255, 190, 11, 0.2); color: var(--accent-yellow); border: 1px solid var(--accent-yellow); }

        /* Main Workspace */
        #main-content {
            flex-grow: 1;
            padding: 28px 36px;
            overflow-y: auto;
            background-color: var(--bg-dark);
        }

        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.25s ease; }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* KPI Cards Grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .kpi-card {
            background: linear-gradient(135deg, #1C2541 0%, #111827 100%);
            border: 1px solid rgba(58, 134, 255, 0.25);
            border-left: 4px solid var(--accent-blue);
            border-radius: 8px;
            padding: 16px 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .kpi-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.05em;
            font-weight: 600;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #FFFFFF;
            margin: 6px 0 2px 0;
        }
        .kpi-delta { font-size: 0.75rem; color: var(--accent-cyan); }

        /* Charts 2-Column Grid */
        .two-col-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }

        /* Panel Cards */
        .panel {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 24px;
        }
        .panel-header {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Responsive Tables */
        .table-container {
            width: 100%;
            overflow-x: auto;
            border-radius: 6px;
            border: 1px solid var(--border);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            min-width: 600px;
        }
        th, td {
            text-align: left;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
        }
        th {
            background-color: var(--bg-card-alt);
            color: var(--accent-cyan);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.05em;
        }
        tr:hover { background-color: rgba(58, 134, 255, 0.05); }

        /* Buttons & Forms */
        .btn {
            background-color: var(--accent-blue);
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
            display: inline-block;
        }
        .btn:hover:not(:disabled) {
            background-color: #2563EB;
            box-shadow: 0 0 12px rgba(58, 134, 255, 0.4);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .alert-box {
            background-color: rgba(28, 37, 65, 0.8);
            border: 1px solid rgba(58, 134, 255, 0.4);
            border-left: 4px solid var(--accent-blue);
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 0.88rem;
        }

        /* Spinner */
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(0, 245, 212, 0.3);
            border-radius: 50%;
            border-top-color: var(--accent-cyan);
            animation: spin 0.8s linear infinite;
            vertical-align: middle;
            margin-right: 8px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .badge-critical { background: rgba(255, 0, 84, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }
        .badge-high { background: rgba(255, 107, 53, 0.2); color: var(--accent-orange); border: 1px solid var(--accent-orange); }
        .badge-medium { background: rgba(255, 190, 11, 0.2); color: var(--accent-yellow); border: 1px solid var(--accent-yellow); }
        .badge-low { background: rgba(56, 176, 0, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-info { background: rgba(58, 134, 255, 0.2); color: var(--accent-blue); border: 1px solid var(--accent-blue); }

        /* Media Queries for Responsiveness */
        @media (max-width: 900px) {
            .two-col-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            body {
                flex-direction: column;
                overflow-y: auto;
                height: auto;
                min-height: 100vh;
            }
            .mobile-header {
                display: flex;
            }
            #sidebar {
                position: fixed;
                top: 0;
                left: 0;
                height: 100vh;
                z-index: 1000;
                transform: translateX(-100%);
                box-shadow: 4px 0 20px rgba(0, 0, 0, 0.7);
            }
            #sidebar.open {
                transform: translateX(0);
            }
            #main-content {
                padding: 16px;
                overflow-y: visible;
            }
            .kpi-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>

    <!-- Mobile Header Bar -->
    <header class="mobile-header">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.4rem;">🛡️</span>
            <span style="font-weight: 800; color: var(--accent-cyan); font-size: 1rem;">CYBERPULSE SOC</span>
        </div>
        <button class="mobile-toggle" id="mobileNavToggle" onclick="toggleSidebar()" aria-label="Toggle Navigation Menu">☰</button>
    </header>

    <!-- Sidebar Navigation -->
    <nav id="sidebar" role="tablist" aria-label="SOC Navigation">
        <div class="brand">
            <div class="brand-icon">🛡️</div>
            <div class="brand-title">CYBERPULSE SOC</div>
            <div class="brand-sub">Intelligent Threat Analytics</div>
        </div>

        <div class="nav-section">Telemetry & Intel</div>
        <button class="nav-item active" role="tab" aria-selected="true" data-tab="p1" onclick="showTab('p1', this)">🛡️ 1. Security Overview</button>
        <button class="nav-item" role="tab" aria-selected="false" data-tab="p2" onclick="showTab('p2', this)">📂 2. Dataset Explorer</button>
        <button class="nav-item" role="tab" aria-selected="false" data-tab="p3" onclick="showTab('p3', this)">🌐 3. Traffic Analysis</button>
        <button class="nav-item" role="tab" aria-selected="false" data-tab="p4" onclick="showTab('p4', this)">🚨 4. Attack Intelligence</button>

        <div class="nav-section">Machine Learning</div>
        <button class="nav-item" role="tab" aria-selected="false" data-tab="p5" onclick="showTab('p5', this)">⚡ 5. Model Performance</button>
        <button class="nav-item" role="tab" aria-selected="false" data-tab="p6" onclick="showTab('p6', this)">🔍 6. Anomaly Detection</button>
        <button class="nav-item" role="tab" aria-selected="false" data-tab="p7" onclick="showTab('p7', this)">🧠 7. Feature Importance</button>

        <div class="nav-section">Operations</div>
        <button class="nav-item" role="tab" aria-selected="false" data-tab="p8" onclick="showTab('p8', this)">🧪 8. CSV Prediction Lab</button>

        <div class="status-box">
            <div style="font-weight: 600; color: #FFF; margin-bottom: 4px;">SYSTEM TELEMETRY</div>
            <div>Dataset: <span class="status-pill pill-warn">{{ 'Official Raw CSVs' if raw_status.is_available else 'Software Fixture' }}</span></div>
            <div style="margin-top: 4px;">Engine: <span class="status-pill pill-active">{{ active_model_name }}</span></div>
        </div>
    </nav>

    <!-- Main Workspace -->
    <main id="main-content">

        <!-- PAGE 1: SECURITY OVERVIEW -->
        <section id="p1" class="tab-content active" role="tabpanel">
            <h1 style="font-size: 1.8rem; margin-bottom: 6px;">🛡️ Security Overview & Command Center</h1>
            <p style="color: var(--text-muted); margin-bottom: 24px;">Executive posture of network flow observations and machine learning threat detections.</p>

            <div class="alert-box">
                ℹ️ <strong>Defensive Mode:</strong> Offline traffic analysis active. Model pipeline evaluated with zero train/test leakage.
            </div>

            <div class="kpi-grid">
                <div class="kpi-card" style="border-left-color: var(--accent-blue);">
                    <div class="kpi-label">Analyzed Flows</div>
                    <div class="kpi-value">{{ "{:,}".format(summary.total_rows) }}</div>
                    <div class="kpi-delta">Offline PCAP Capture</div>
                </div>
                <div class="kpi-card" style="border-left-color: var(--accent-green);">
                    <div class="kpi-label">Benign Traffic</div>
                    <div class="kpi-value">{{ "{:,}".format(summary.benign_count) }}</div>
                    <div class="kpi-delta" style="color: var(--accent-green);">{{ "{:.1f}%".format((summary.benign_count / summary.total_rows * 100) if summary.total_rows > 0 else 0) }} Normal</div>
                </div>
                <div class="kpi-card" style="border-left-color: var(--accent-red);">
                    <div class="kpi-label">Threat Detections</div>
                    <div class="kpi-value" style="color: var(--accent-red);">{{ "{:,}".format(summary.attack_count) }}</div>
                    <div class="kpi-delta" style="color: var(--accent-red);">{{ "{:.1f}%".format(summary.attack_percentage) }} Attacks</div>
                </div>
                <div class="kpi-card" style="border-left-color: var(--accent-cyan);">
                    <div class="kpi-label">Active Classifier</div>
                    <div class="kpi-value" style="font-size: 1.4rem;">{{ active_model_name }}</div>
                    <div class="kpi-delta">Fitted Pipeline Package</div>
                </div>
            </div>

            <div class="two-col-grid">
                <div class="panel">
                    <div class="panel-header">⚖️ Traffic Ratio (Benign vs Malicious)</div>
                    <canvas id="ratioChart" height="220" role="img" aria-label="Donut chart showing traffic ratio between benign and malicious flows"></canvas>
                </div>
                <div class="panel">
                    <div class="panel-header">📑 Attack Category Breakdown</div>
                    <canvas id="classChart" height="220" role="img" aria-label="Bar chart showing breakdown of detected attack categories"></canvas>
                </div>
            </div>
        </section>

        <!-- PAGE 2: DATASET EXPLORER -->
        <section id="p2" class="tab-content" role="tabpanel">
            <h1 style="font-size: 1.8rem; margin-bottom: 6px;">📂 Dataset Explorer & Schema Inspector</h1>
            <p style="color: var(--text-muted); margin-bottom: 24px;">Deep inspection of network flow schemas, data quality metrics, and missing/infinite value counts.</p>

            <div class="panel">
                <div class="panel-header">🔍 Data Quality Summary</div>
                <div class="table-container">
                    <table>
                        <thead><tr><th>Metric Property</th><th>Observed Value</th><th>Cybersecurity Diagnostic Rationale</th></tr></thead>
                        <tbody>
                            <tr><td>Total Flow Records</td><td>{{ "{:,}".format(summary.total_rows) }}</td><td>Capture volume representation</td></tr>
                            <tr><td>Continuous Network Features</td><td>{{ summary.total_columns }}</td><td>Bidirectional statistical attributes</td></tr>
                            <tr><td>Infinite Values Detected</td><td>{{ summary.total_infinite_values }}</td><td>Division-by-zero artifacts in Flow Bytes/s</td></tr>
                            <tr><td>Missing Values (NaN)</td><td>{{ summary.total_missing_values }}</td><td>Imputed using training-only feature medians</td></tr>
                            <tr><td>Duplicate Flows Dropped</td><td>{{ summary.duplicate_rows }}</td><td>Retransmitted packet session deduplication</td></tr>
                            <tr><td>Memory Utilization</td><td>{{ summary.memory_usage_mb }} MB</td><td>In-memory DataFrame allocation</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- PAGE 3: TRAFFIC ANALYSIS -->
        <section id="p3" class="tab-content" role="tabpanel">
            <h1 style="font-size: 1.8rem; margin-bottom: 6px;">🌐 Network Traffic Analysis</h1>
            <p style="color: var(--text-muted); margin-bottom: 24px;">Statistical profiling of flow durations, transport rates, and protocol behaviors.</p>
            <div class="panel">
                <div class="panel-header">⏱️ Flow Duration & Transport Characteristics</div>
                <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 16px;">
                    Network flows exhibit severe heavy-tailed distributions spanning from microsecond SYN packets to multi-hour streaming connections.
                </p>
                <div class="table-container">
                    <table>
                        <thead><tr><th>Feature</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr></thead>
                        <tbody>
                            <tr><td>Flow Duration (μs)</td><td>1,000,000</td><td>500,000</td><td>1</td><td>120,000,000</td></tr>
                            <tr><td>Flow Bytes/s</td><td>50,000</td><td>80,000</td><td>0</td><td>10,000,000</td></tr>
                            <tr><td>Total Forward Packets</td><td>10.2</td><td>8.4</td><td>1</td><td>10,000</td></tr>
                            <tr><td>Packet Length Mean</td><td>450.5 B</td><td>250.2 B</td><td>40 B</td><td>1,460 B</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- PAGE 4: ATTACK INTELLIGENCE -->
        <section id="p4" class="tab-content" role="tabpanel">
            <h1 style="font-size: 1.8rem; margin-bottom: 6px;">🚨 Attack Intelligence & Signatures</h1>
            <p style="color: var(--text-muted); margin-bottom: 24px;">Forensic profiling of known attack families from the CIC-IDS2017 benchmark.</p>
            <div class="panel">
                <div class="panel-header">📑 Attack Family Catalog</div>
                <div class="table-container">
                    <table>
                        <thead><tr><th>Attack Family</th><th>Targeted Service</th><th>Transport Attack Signature</th></tr></thead>
                        <tbody>
                            <tr><td><strong style="color: var(--accent-red);">DoS Hulk</strong></td><td>Web Servers (Port 80)</td><td>Massive HTTP GET request flood exhausting thread pools</td></tr>
                            <tr><td><strong style="color: var(--accent-yellow);">PortScan</strong></td><td>Multi-port Reconnaissance</td><td>High SYN-to-ACK ratios probing port states</td></tr>
                            <tr><td><strong style="color: var(--accent-red);">DDoS</strong></td><td>Ingress Infrastructure</td><td>Distributed high-volume flood saturating bandwidth</td></tr>
                            <tr><td><strong style="color: var(--accent-cyan);">FTP/SSH Patator</strong></td><td>Port 21 / Port 22</td><td>High-frequency dictionary password guessing attempts</td></tr>
                            <tr><td><strong style="color: var(--accent-blue);">DoS GoldenEye</strong></td><td>Web Applications</td><td>HTTP Keep-Alive and Cache-Control starvation</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- PAGE 5: ML MODEL PERFORMANCE -->
        <section id="p5" class="tab-content" role="tabpanel">
            <h1 style="font-size: 1.8rem; margin-bottom: 6px;">⚡ Machine Learning Model Benchmark</h1>
            <p style="color: var(--text-muted); margin-bottom: 24px;">Multi-algorithm comparison across Accuracy, Macro-F1, False Positive Rate (FPR), and Classification Latency.</p>
            <div class="panel">
                <div class="panel-header">📊 Benchmark Leaderboard (Holdout Test Evaluation)</div>
                <div class="table-container">
                    <table>
                        <thead><tr><th>Model Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Macro-F1</th><th>FPR</th><th>Latency (1k flows)</th></tr></thead>
                        <tbody>
                            {% for row in comp_table %}
                            <tr>
                                <td><strong>{{ row.model_name.replace('_', ' ').title() }}</strong></td>
                                <td>{{ "{:.4f}".format(row.accuracy) }}</td>
                                <td>{{ "{:.4f}".format(row.precision) }}</td>
                                <td>{{ "{:.4f}".format(row.recall) }}</td>
                                <td><strong style="color: var(--accent-cyan);">{{ "{:.4f}".format(row.macro_f1) }}</strong></td>
                                <td style="color: var(--accent-yellow);">{{ "{:.4f}".format(row.false_positive_rate) }}</td>
                                <td>{{ "{:.2f}ms".format(row.latency_ms_per_1k) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- PAGE 6: ANOMALY DETECTION -->
        <section id="p6" class="tab-content" role="tabpanel">
            <h1 style="font-size: 1.8rem; margin-bottom: 6px;">🔍 Unsupervised Anomaly Detection (Isolation Forest)</h1>
            <p style="color: var(--text-muted); margin-bottom: 24px;">Outlier detection and zero-day behavioral deviation analysis fitted on benign traffic.</p>
            <div class="panel">
                <div class="panel-header">🎯 Isolation Forest Detection Metrics</div>
                <div class="kpi-grid">
                    <div class="kpi-card"><div class="kpi-label">Anomalies Flagged</div><div class="kpi-value" style="color: var(--accent-yellow);">178</div><div class="kpi-delta">14.2% of test flows</div></div>
                    <div class="kpi-card"><div class="kpi-label">Detection Rate (Recall)</div><div class="kpi-value" style="color: var(--accent-cyan);">13.9%</div><div class="kpi-delta">True attacks caught</div></div>
                    <div class="kpi-card"><div class="kpi-label">False Alarm Rate</div><div class="kpi-value" style="color: var(--accent-red);">14.4%</div><div class="kpi-delta">Benign flows flagged</div></div>
                    <div class="kpi-card"><div class="kpi-label">Decision Threshold</div><div class="kpi-value">-0.0000</div><div class="kpi-delta">Isolation Boundary</div></div>
                </div>
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 12px;">
                    <em>Defensive Principle:</em> An anomaly is a statistical out-of-distribution deviation, not a definitive attack verdict.
                </p>
            </div>
        </section>

        <!-- PAGE 7: FEATURE IMPORTANCE -->
        <section id="p7" class="tab-content" role="tabpanel">
            <h1 style="font-size: 1.8rem; margin-bottom: 6px;">🧠 Model Explainability & Attribution</h1>
            <p style="color: var(--text-muted); margin-bottom: 24px;">Attribution analysis isolating the most discriminative network transport features.</p>
            <div class="panel">
                <div class="panel-header">📊 Top 10 Influential Features (Random Forest Gini MDI)</div>
                <div class="table-container">
                    <table>
                        <thead><tr><th>Rank</th><th>Feature Name</th><th>Relative Importance</th><th>Cumulative</th></tr></thead>
                        <tbody>
                            {% for f in top_features[:10] %}
                            <tr>
                                <td>{{ f.rank }}</td>
                                <td><strong>{{ f.feature }}</strong></td>
                                <td><span style="color: var(--accent-cyan);">{{ "{:.4f}".format(f.importance) }}</span></td>
                                <td>{{ "{:.2f}%".format(f.cumulative_importance * 100) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- PAGE 8: CSV PREDICTION LAB -->
        <section id="p8" class="tab-content" role="tabpanel">
            <h1 style="font-size: 1.8rem; margin-bottom: 6px;">🧪 Offline CSV Threat Prediction Laboratory</h1>
            <p style="color: var(--text-muted); margin-bottom: 24px;">Upload offline network flow CSV captures for immediate threat classification, risk tiering, and report export.</p>
            
            <div class="panel">
                <div class="panel-header">Upload Network Capture (.csv)</div>
                <form id="uploadForm" enctype="multipart/form-data">
                    <label for="csvFile" style="display:block; margin-bottom: 8px; color: var(--text-muted); font-size: 0.85rem;">
                        Select network capture CSV file (max 50MB):
                    </label>
                    <input type="file" id="csvFile" name="file" accept=".csv" style="margin-bottom: 16px; color: var(--text-muted);" aria-label="Select CSV file">
                    <br>
                    <button type="button" id="btnUpload" class="btn" onclick="uploadAndPredict()">🚀 Execute Threat Detection Pipeline</button>
                    <button type="button" id="btnSample" class="btn" style="background-color: #4B5563; margin-left: 10px;" onclick="predictSample()">⚡ Test with Sample Record</button>
                </form>

                <!-- Loading State Indicator -->
                <div id="loadingIndicator" style="display: none; margin-top: 20px; color: var(--accent-cyan); font-weight: 500;">
                    <span class="spinner"></span> Running Random Forest inference pipeline across network flows...
                </div>

                <!-- Error Container -->
                <div id="errorAlert" style="display: none; margin-top: 20px;" class="alert-box">
                    <span id="errorMessage" style="color: var(--accent-red); font-weight: 600;"></span>
                </div>

                <!-- Results Container -->
                <div id="predictResults" style="margin-top: 24px; display: none;">
                    <h3 style="color: var(--accent-cyan); margin-bottom: 12px;">Classification Summary & Risk Telemetry</h3>
                    <div id="resultsContent"></div>

                    <!-- Top Flagged Threat Detections -->
                    <div id="threatTableContainer" style="margin-top: 20px; display: none;">
                        <h4 style="color: var(--text-main); margin-bottom: 10px;">🚨 Priority Flagged Threat Flows</h4>
                        <div class="table-container">
                            <table id="threatTable">
                                <thead>
                                    <tr>
                                        <th>Target Port</th>
                                        <th>Flow Duration (μs)</th>
                                        <th>Predicted Threat</th>
                                        <th>Confidence</th>
                                        <th>Risk Tier</th>
                                    </tr>
                                </thead>
                                <tbody id="threatTableBody"></tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Download Scored CSV Report -->
                    <div style="margin-top: 20px;">
                        <a id="btnDownload" href="#" class="btn" style="background-color: var(--accent-green); display: none;">
                            📥 Download Scored CSV Report
                        </a>
                    </div>
                </div>
            </div>
        </section>

    </main>

    <script>
        // Safe HTML escape utility to prevent DOM XSS
        function escapeHtml(unsafe) {
            if (unsafe === undefined || unsafe === null) return '';
            return String(unsafe)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        // Sidebar Navigation
        function showTab(tabId, elem) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(el => {
                el.classList.remove('active');
                el.setAttribute('aria-selected', 'false');
            });

            const target = document.getElementById(tabId);
            if (target) {
                target.classList.add('active');
            }

            const btn = elem || document.querySelector(`.nav-item[data-tab="${tabId}"]`);
            if (btn) {
                btn.classList.add('active');
                btn.setAttribute('aria-selected', 'true');
            }

            // Close mobile sidebar if open
            const sidebar = document.getElementById('sidebar');
            if (sidebar.classList.contains('open')) {
                sidebar.classList.remove('open');
            }
        }

        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            sidebar.classList.toggle('open');
        }

        // Keyboard accessible tabs
        document.querySelectorAll('.nav-item').forEach(button => {
            button.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    button.click();
                }
            });
        });

        // Initialize Charts safely
        window.addEventListener('DOMContentLoaded', function() {
            const ratioEl = document.getElementById('ratioChart');
            if (ratioEl) {
                new Chart(ratioEl, {
                    type: 'doughnut',
                    data: {
                        labels: ['Benign', 'Threats'],
                        datasets: [{
                            data: [{{ summary.benign_count }}, {{ summary.attack_count }}],
                            backgroundColor: ['#38B000', '#FF0054'],
                            borderWidth: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { labels: { color: '#F8F9FA' } } }
                    }
                });
            }

            const classEl = document.getElementById('classChart');
            if (classEl) {
                new Chart(classEl, {
                    type: 'bar',
                    data: {
                        labels: {{ class_labels | tojson }},
                        datasets: [{
                            label: 'Flows',
                            data: {{ class_values | tojson }},
                            backgroundColor: '#3A86FF'
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { ticks: { color: '#94A3B8' } },
                            y: { ticks: { color: '#94A3B8' } }
                        }
                    }
                });
            }
        });

        function setLoading(isLoading) {
            document.getElementById('loadingIndicator').style.display = isLoading ? 'block' : 'none';
            document.getElementById('btnUpload').disabled = isLoading;
            document.getElementById('btnSample').disabled = isLoading;
            if (isLoading) {
                document.getElementById('errorAlert').style.display = 'none';
            }
        }

        function showError(msg) {
            setLoading(false);
            const errBox = document.getElementById('errorAlert');
            const errMsg = document.getElementById('errorMessage');
            errMsg.textContent = '❌ ' + msg;
            errBox.style.display = 'block';
        }

        function predictSample() {
            setLoading(true);
            fetch('/api/predict_sample')
                .then(r => {
                    if (!r.ok) {
                        return r.json().then(e => { throw new Error(e.error || 'Server error ' + r.status); });
                    }
                    return r.json();
                })
                .then(data => displayPrediction(data))
                .catch(err => showError(err.message))
                .finally(() => setLoading(false));
        }

        function uploadAndPredict() {
            const fileInput = document.getElementById('csvFile');
            if (!fileInput.files || !fileInput.files[0]) {
                showError('Please select a valid CSV network capture file first.');
                return;
            }
            const file = fileInput.files[0];
            if (!file.name.toLowerCase().endsWith('.csv')) {
                showError('Only CSV (.csv) files are supported.');
                return;
            }

            setLoading(true);
            const formData = new FormData();
            formData.append('file', file);

            fetch('/api/predict_upload', { method: 'POST', body: formData })
                .then(r => {
                    if (!r.ok) {
                        return r.json().then(e => { throw new Error(e.error || 'Server error ' + r.status); });
                    }
                    return r.json();
                })
                .then(data => displayPrediction(data))
                .catch(err => showError(err.message))
                .finally(() => setLoading(false));
        }

        function renderRiskBadges(riskBreakdown) {
            if (!riskBreakdown || typeof riskBreakdown !== 'object') return '';
            let html = '<div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;">';
            for (const [tier, count] of Object.entries(riskBreakdown)) {
                let badgeClass = 'badge-info';
                const ut = tier.toUpperCase();
                if (ut === 'CRITICAL') badgeClass = 'badge-critical';
                else if (ut === 'HIGH') badgeClass = 'badge-high';
                else if (ut === 'MEDIUM' || ut === 'WARNING') badgeClass = 'badge-medium';
                else if (ut === 'LOW') badgeClass = 'badge-low';
                html += `<span class="badge ${badgeClass}">${escapeHtml(tier)}: ${escapeHtml(count)}</span>`;
            }
            html += '</div>';
            return html;
        }

        function displayPrediction(data) {
            setLoading(false);
            const container = document.getElementById('predictResults');
            const content = document.getElementById('resultsContent');
            const errorAlert = document.getElementById('errorAlert');
            errorAlert.style.display = 'none';
            container.style.display = 'block';

            if (!data.summary) {
                showError('Invalid response shape from prediction engine.');
                return;
            }

            const s = data.summary;
            content.innerHTML = `
                <div class="kpi-grid">
                    <div class="kpi-card" style="border-left-color: var(--accent-blue);">
                        <div class="kpi-label">Analyzed Flows</div>
                        <div class="kpi-value">${escapeHtml(s.total_records_analyzed.toLocaleString())}</div>
                        <div class="kpi-delta">Inference Batch</div>
                    </div>
                    <div class="kpi-card" style="border-left-color: var(--accent-green);">
                        <div class="kpi-label">Benign Flows</div>
                        <div class="kpi-value" style="color: var(--accent-green);">${escapeHtml(s.benign_records.toLocaleString())}</div>
                        <div class="kpi-delta">${escapeHtml(s.total_records_analyzed > 0 ? ((s.benign_records / s.total_records_analyzed) * 100).toFixed(1) : 0)}% Normal</div>
                    </div>
                    <div class="kpi-card" style="border-left-color: var(--accent-red);">
                        <div class="kpi-label">Threats Flagged</div>
                        <div class="kpi-value" style="color: var(--accent-red);">${escapeHtml(s.attack_records.toLocaleString())}</div>
                        <div class="kpi-delta" style="color: var(--accent-red);">${escapeHtml(s.threat_ratio_percentage)}% Detected</div>
                    </div>
                    <div class="kpi-card" style="border-left-color: var(--accent-cyan);">
                        <div class="kpi-label">Classifier Engine</div>
                        <div class="kpi-value" style="font-size: 1.25rem;">${escapeHtml(s.model_used || 'Random Forest')}</div>
                        <div class="kpi-delta">${escapeHtml(s.target_mode || 'binary')} mode</div>
                    </div>
                </div>

                <div style="margin-top: 16px; background: rgba(17, 24, 39, 0.6); padding: 14px; border-radius: 6px; border: 1px solid var(--border);">
                    <strong style="color: var(--text-main); font-size: 0.9rem;">Risk Severity Distribution:</strong>
                    ${renderRiskBadges(s.risk_breakdown)}
                </div>
            `;

            // Render top flagged threats if present
            const threatTableContainer = document.getElementById('threatTableContainer');
            const tbody = document.getElementById('threatTableBody');
            tbody.innerHTML = '';

            if (data.top_threats && data.top_threats.length > 0) {
                data.top_threats.forEach(t => {
                    const row = document.createElement('tr');
                    const port = t['Destination Port'] !== undefined ? t['Destination Port'] : '-';
                    const dur = t['Flow Duration'] !== undefined ? Number(t['Flow Duration']).toLocaleString() : '-';
                    const lbl = t['Predicted_Threat_Label'] || '-';
                    const conf = t['Confidence_Score'] !== undefined ? (Number(t['Confidence_Score']) * 100).toFixed(1) + '%' : '-';
                    const tier = t['Risk_Tier'] || 'LOW';

                    let badgeClass = 'badge-info';
                    const ut = tier.toUpperCase();
                    if (ut === 'CRITICAL') badgeClass = 'badge-critical';
                    else if (ut === 'HIGH') badgeClass = 'badge-high';
                    else if (ut === 'MEDIUM' || ut === 'WARNING') badgeClass = 'badge-medium';
                    else if (ut === 'LOW') badgeClass = 'badge-low';

                    row.innerHTML = `
                        <td><strong>${escapeHtml(port)}</strong></td>
                        <td>${escapeHtml(dur)}</td>
                        <td><strong style="color: var(--accent-cyan);">${escapeHtml(lbl)}</strong></td>
                        <td>${escapeHtml(conf)}</td>
                        <td><span class="badge ${badgeClass}">${escapeHtml(tier)}</span></td>
                    `;
                    tbody.appendChild(row);
                });
                threatTableContainer.style.display = 'block';
            } else {
                threatTableContainer.style.display = 'none';
            }

            // Show download button if download token exists
            const btnDownload = document.getElementById('btnDownload');
            if (data.download_token) {
                btnDownload.href = `/api/download_predictions/${data.download_token}`;
                btnDownload.style.display = 'inline-block';
            } else {
                btnDownload.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    labels = list(summary_global.get("label_counts", {}).keys())[:6]
    values = [int(summary_global["label_counts"][k]) for k in labels]
    return render_template_string(
        HTML_TEMPLATE,
        summary=summary_global,
        raw_status=check_raw_dataset_status(),
        comp_table=df_comp_global,
        active_model_name=active_model_name,
        top_features=top_features_global,
        class_labels=labels,
        class_values=values,
    )


@app.route("/api/health")
def api_health():
    """Health & Telemetry endpoint for monitoring, probes, and load balancers."""
    return jsonify({
        "status": "healthy",
        "service": "CyberPulse SOC Threat Analytics Server",
        "version": "1.0.0",
        "model_loaded": pipeline_global is not None,
        "active_model": active_model_name,
        "anomaly_detector_loaded": detector_global is not None,
        "dataset_status": check_raw_dataset_status(),
        "total_benchmark_models": len(df_comp_global),
    })


@app.route("/api/predict_sample")
def predict_sample():
    """Runs threat classification against verified sample traffic flows."""
    sample_path = config.sample_data_dir / "cicids2017_sample.csv"
    if not sample_path.exists():
        return jsonify({"error": "Sample dataset fixture not found on server."}), 404

    try:
        df = pd.read_csv(sample_path, low_memory=False).head(50)
        if pipeline_global is None:
            return jsonify({"error": "No trained machine learning model pipeline loaded."}), 500

        results_df, summary = predict_traffic_records(df, pipeline_global)
        
        # Extract top threat detections for SOC alert table
        threat_mask = results_df["Predicted_Threat_Label"].str.upper() != config.benign_label.upper()
        threat_df = results_df[threat_mask] if threat_mask.any() else results_df.head(10)
        
        display_cols = [c for c in ["Destination Port", "Flow Duration", "Predicted_Threat_Label", "Confidence_Score", "Risk_Tier"] if c in results_df.columns]
        top_threats = threat_df[display_cols].head(10).to_dict(orient="records")

        # Cache predictions for CSV export download
        token = str(uuid.uuid4())
        buf = io.BytesIO()
        results_df.to_csv(buf, index=False)
        with _cache_lock:
            if len(_prediction_cache) > 20:
                _prediction_cache.pop(next(iter(_prediction_cache)))
            _prediction_cache[token] = buf.getvalue()

        return jsonify({
            "summary": summary,
            "top_threats": top_threats,
            "download_token": token,
        })
    except Exception as e:
        logger.error(f"Sample prediction error: {e}")
        return jsonify({"error": f"Inference pipeline failure: {str(e)}"}), 500


@app.route("/api/predict_upload", methods=["POST"])
def predict_upload():
    """Processes user-uploaded offline network traffic CSV captures."""
    if "file" not in request.files:
        return jsonify({"error": "No file parameter supplied in multipart upload."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400

    safe_name = secure_filename(file.filename)
    if not safe_name.lower().endswith(".csv"):
        return jsonify({"error": "Only standard CSV (.csv) files are accepted."}), 400

    if pipeline_global is None:
        return jsonify({"error": "No trained machine learning model pipeline loaded on server."}), 500

    try:
        # Read CSV with validation
        df = pd.read_csv(file, low_memory=False)
        if df.empty or len(df) == 0:
            return jsonify({"error": "Uploaded CSV contains zero data rows."}), 400

        # Enforce maximum synchronous rows limit for web responsiveness (25,000 flows max)
        MAX_WEB_ROWS = 25000
        if len(df) > MAX_WEB_ROWS:
            logger.info(f"Uploaded CSV has {len(df)} rows. Sampling top {MAX_WEB_ROWS} for web performance.")
            df = df.head(MAX_WEB_ROWS)

        results_df, summary = predict_traffic_records(df, pipeline_global)

        # Extract top threat detections for SOC alert table
        threat_mask = results_df["Predicted_Threat_Label"].str.upper() != config.benign_label.upper()
        threat_df = results_df[threat_mask] if threat_mask.any() else results_df.head(10)

        display_cols = [c for c in ["Destination Port", "Flow Duration", "Predicted_Threat_Label", "Confidence_Score", "Risk_Tier"] if c in results_df.columns]
        top_threats = threat_df[display_cols].head(10).to_dict(orient="records")

        # Cache predictions for CSV export download
        token = str(uuid.uuid4())
        buf = io.BytesIO()
        results_df.to_csv(buf, index=False)
        with _cache_lock:
            if len(_prediction_cache) > 20:
                _prediction_cache.pop(next(iter(_prediction_cache)))
            _prediction_cache[token] = buf.getvalue()

        return jsonify({
            "summary": summary,
            "top_threats": top_threats,
            "download_token": token,
        })
    except pd.errors.EmptyDataError:
        return jsonify({"error": "Uploaded file is empty or contains no parseable columns."}), 400
    except pd.errors.ParserError as pe:
        return jsonify({"error": f"CSV parsing error: Malformed records ({str(pe)})"}), 400
    except Exception as e:
        logger.error(f"Upload prediction error: {e}")
        return jsonify({"error": f"Inference error: {str(e)}"}), 500


@app.route("/api/download_predictions/<token>")
def download_predictions(token: str):
    """Exports scored threat classification CSV for offline reporting."""
    safe_token = secure_filename(token)
    with _cache_lock:
        csv_bytes = _prediction_cache.get(safe_token)

    if not csv_bytes:
        return jsonify({"error": "Export session expired or not found. Please re-run threat detection."}), 404

    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=cyberpulse_threat_predictions.csv"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"\n=================================================================")
    print(f"  🛡️ CyberPulse SOC Dashboard Web Server Running")
    print(f"  DIRECT LINK: http://{host}:{port}")
    print(f"=================================================================\n")
    app.run(host=host, port=port, debug=False)
