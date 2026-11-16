"""
Page 1: Security Overview & SOC Executive Dashboard
"""

import streamlit as st
import pandas as pd
from dashboard.components.metrics import render_metric_card
from src.visualization import get_plotly_benign_vs_attack, get_plotly_class_distribution


def render_page(df: pd.DataFrame, dataset_meta: dict, eval_metrics: dict, selected_model_name: str):
    st.markdown("## 🛡️ Security Overview & SOC Command Center")
    st.markdown("Real-time executive posture of network flow observations and machine learning threat detections.")

    # Status notice if in software verification mode
    if dataset_meta.get("is_sample"):
        st.warning(
            "⚠️ **Software Verification Mode Active**: Real multi-gigabyte CIC-IDS2017 files are not yet in `data/raw/`. "
            "Currently analyzing verified sample test schema. See **Dataset Explorer** for download instructions."
        )

    # Top KPI Metrics Cards
    total_flows = len(df)
    target_col = "Label" if "Label" in df.columns else df.columns[-1]

    benign_cnt = int((df[target_col].astype(str).str.strip().str.upper() == "BENIGN").sum())
    attack_cnt = total_flows - benign_cnt
    attack_pct = round((attack_cnt / total_flows * 100), 2) if total_flows > 0 else 0.0
    num_classes = df[target_col].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card("Analyzed Flows", f"{total_flows:,}", "Offline capture", "normal", "#3A86FF", "📊"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("Benign Traffic", f"{benign_cnt:,}", f"{100-attack_pct:.1f}% normal", "success", "#38B000", "✅"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card("Threat Detections", f"{attack_cnt:,}", f"{attack_pct:.1f}% attacks", "danger", "#FF0054", "🚨"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card("Active Algorithm", selected_model_name or "Random Forest", "Fitted pipeline", "normal", "#00F5D4", "⚡"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### ⚖️ Threat Traffic Distribution")
        fig_donut = get_plotly_benign_vs_attack(df, target_col=target_col)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown("#### 📑 Attack Category Breakdown")
        fig_classes = get_plotly_class_distribution(df, target_col=target_col)
        st.plotly_chart(fig_classes, use_container_width=True)

    # Model Performance Summary Callout
    st.markdown("---")
    st.markdown("#### 🎯 Machine Learning Threat Engine Summary")
    st.info(
        f"**Defensive Threat Engine Status:** The system uses calibrated tree ensembles (Random Forest, XGBoost) "
        f"and linear baselines trained on CICFlowMeter bidirectional statistical flows. "
        f"Current model selection metric is **Macro-F1**, prioritizing high detection across minority attack vectors."
    )

