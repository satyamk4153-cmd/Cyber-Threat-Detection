"""
Sidebar component for Streamlit SOC application.
Renders navigation menu, system status, dataset status, and project metadata.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

from src.config import config
from src.data_loader import check_raw_dataset_status


def render_sidebar(available_models: List[str]) -> Dict[str, Any]:
    """Renders the SOC sidebar and returns user selections."""
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 12px 0 20px 0;">
            <div style="font-size: 2.2rem; margin-bottom: 4px;">🛡️</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #00F5D4; letter-spacing: 0.05em;">
                CYBERPULSE SOC
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em;">
                Intelligent Threat Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size: 0.78rem; font-weight: 700; color: #3A86FF; text-transform: uppercase;'>Navigation</div>",
        unsafe_allow_html=True,
    )

    pages = [
        "🛡️ 1. Security Overview",
        "📂 2. Dataset Explorer",
        "🌐 3. Network Traffic Analysis",
        "🚨 4. Attack Intelligence",
        "⚡ 5. ML Model Performance",
        "🔍 6. Anomaly Detection (IsoForest)",
        "🧠 7. Feature Importance & Explainability",
        "🧪 8. CSV Prediction Lab",
        "📘 9. About & Academic Project",
    ]

    selected_page = st.sidebar.radio("Go to:", pages, index=0, label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size: 0.78rem; font-weight: 700; color: #3A86FF; text-transform: uppercase;'>SOC Telemetry Status</div>",
        unsafe_allow_html=True,
    )

    # Check raw dataset status
    raw_status = check_raw_dataset_status()
    if raw_status["is_available"]:
        st.sidebar.success(f"● Official Dataset: {raw_status['file_count']} files ({raw_status['total_size_mb']:.1f} MB)")
    else:
        st.sidebar.warning("● Mode: Software Verification Fixture")
        st.sidebar.caption("CIC-IDS2017 raw files not in data/raw/. Running on verified sample schema.")

    # Check trained models status
    model_count = len(available_models)
    if model_count > 0:
        st.sidebar.info(f"● Active Models: {model_count} Loaded")
    else:
        st.sidebar.error("● Active Models: 0 Loaded")

    # Selected Model Picker for Predictions
    selected_model = None
    if available_models:
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "<div style='font-size: 0.78rem; font-weight: 700; color: #00F5D4; text-transform: uppercase;'>Active Inference Model</div>",
            unsafe_allow_html=True,
        )
        selected_model = st.sidebar.selectbox("Model:", available_models, index=0, label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        """
        **Defensive Cyber Research Project**  
        • NPTEL Python for Data Science  
        • B.Tech Computer Science Academic Project  
        • Dataset: UNB CIC-IDS2017
        """
    )

    return {
        "page": selected_page,
        "selected_model": selected_model,
        "raw_status": raw_status,
    }

