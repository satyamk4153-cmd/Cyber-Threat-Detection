"""
Main Streamlit Application: CyberPulse Intelligent Threat Detection SOC
Provides an interactive, dark-themed Security Operations Center interface.
"""

from pathlib import Path
import sys

# Ensure project root is available in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import streamlit as st

from src.config import config
from src.data_loader import load_dataset
from dashboard.components.sidebar import render_sidebar
from dashboard.pages import (
    p1_overview,
    p2_dataset_explorer,
    p3_traffic_analysis,
    p4_attack_intelligence,
    p5_ml_performance,
    p6_anomaly_detection,
    p7_feature_importance,
    p8_prediction_lab,
    p9_about,
)

# Page configuration
st.set_page_config(
    page_title="CyberPulse SOC - Threat Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom SOC Dark Theme CSS styling
st.markdown(
    """
    <style>
        /* Global Body & Backgrounds */
        .stApp {
            background-color: #0B132B;
            color: #F8F9FA;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0D1B2A !important;
            border-right: 1px solid #1C2541;
        }

        /* Metric card styling */
        div[data-testid="stMetricValue"] {
            color: #00F5D4 !important;
            font-weight: 700;
        }

        /* Streamlit Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1C2541;
            border-radius: 6px 6px 0 0;
            padding: 8px 18px;
            color: #94A3B8;
            border: 1px solid #1E293B;
            border-bottom: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E293B !important;
            color: #00F5D4 !important;
            border-top: 2px solid #00F5D4 !important;
        }

        /* Dataframe styling */
        div[data-testid="stDataFrame"] {
            border: 1px solid #1C2541;
            border-radius: 8px;
            background-color: #111827;
        }

        /* Buttons */
        .stButton>button {
            background-color: #3A86FF;
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            padding: 8px 20px;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #2563EB;
            box-shadow: 0 0 12px rgba(58, 134, 255, 0.5);
            color: #FFFFFF;
        }

        /* Success / Info / Warning Callout boxes */
        div.stAlert {
            background-color: #1C2541;
            border: 1px solid #3A86FF44;
            color: #F8F9FA;
            border-radius: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner="Loading network telemetry records...")
def get_cached_dataset():
    """Loads and caches the dataset to optimize UI responsiveness."""
    return load_dataset(use_sample_if_empty=True)


@st.cache_data(ttl=3600)
def get_cached_metrics():
    """Loads model evaluation metrics if present."""
    m_file = config.results_dir / "evaluation_metrics_binary.json"
    if m_file.exists():
        with open(m_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_available_models():
    """Scans models directory for saved pipelines."""
    models = []
    if config.saved_models_dir.exists():
        for f in config.saved_models_dir.glob("*.joblib"):
            if "isolation" not in f.name.lower():
                models.append(f.stem.replace("_", " ").title())
    return models or ["Random Forest Binary", "Decision Tree Binary"]


def main():
    # Load data and models
    try:
        df, dataset_meta = get_cached_dataset()
    except Exception as e:
        st.error(f"Critical error loading dataset: {e}")
        st.stop()

    available_models = get_available_models()
    eval_metrics = get_cached_metrics()

    # Render sidebar navigation
    nav_state = render_sidebar(available_models)
    current_page = nav_state["page"]
    selected_model_name = nav_state["selected_model"]

    # Dispatch to selected page
    if "1. Security Overview" in current_page:
        p1_overview.render_page(df, dataset_meta, eval_metrics, selected_model_name)
    elif "2. Dataset Explorer" in current_page:
        p2_dataset_explorer.render_page(df, dataset_meta)
    elif "3. Network Traffic Analysis" in current_page:
        p3_traffic_analysis.render_page(df)
    elif "4. Attack Intelligence" in current_page:
        p4_attack_intelligence.render_page(df)
    elif "5. ML Model Performance" in current_page:
        p5_ml_performance.render_page()
    elif "6. Anomaly Detection" in current_page:
        p6_anomaly_detection.render_page(df)
    elif "7. Feature Importance" in current_page:
        p7_feature_importance.render_page(df)
    elif "8. CSV Prediction Lab" in current_page:
        p8_prediction_lab.render_page(selected_model_name)
    elif "9. About & Academic Project" in current_page:
        p9_about.render_page()


if __name__ == "__main__":
    main()

