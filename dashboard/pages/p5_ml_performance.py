"""
Page 5: Machine Learning Model Benchmark & Diagnostic Metrics
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from src.config import config
from src.visualization import (
    SOC_PALETTE,
    get_plotly_model_comparison,
    get_plotly_roc_curves,
)


def render_page():
    st.markdown("## ⚡ Machine Learning Benchmark & Diagnostic Evaluation")
    st.markdown("Rigorous multi-model comparison across detection accuracy, false positive rates, and real-time classification latency.")

    results_csv = config.results_dir / "model_comparison_binary.csv"
    metrics_json = config.results_dir / "evaluation_metrics_binary.json"

    if not results_csv.exists() or not metrics_json.exists():
        st.warning("⚠️ Model evaluation results not yet generated. Please execute model training via `python main.py --train`.")
        return

    df_comp = pd.read_csv(results_csv)
    with open(metrics_json, "r", encoding="utf-8") as f:
        metrics_dict = json.load(f)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Benchmark Comparison",
        "🎯 Confusion Matrices",
        "📈 ROC Curves",
        "⏱️ Speed vs Accuracy Tradeoffs",
    ])

    with tab1:
        st.markdown("#### Comprehensive Performance Leaderboard")
        st.dataframe(
            df_comp.style.format({
                "accuracy": "{:.4f}",
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1_score": "{:.4f}",
                "macro_f1": "{:.4f}",
                "weighted_f1": "{:.4f}",
                "false_positive_rate": "{:.4f}",
                "false_negative_rate": "{:.4f}",
                "roc_auc": "{:.4f}",
                "training_time_sec": "{:.3f}s",
                "latency_ms_per_1k": "{:.3f}ms",
            }),
            use_container_width=True,
        )

        st.markdown("#### Multi-Metric Comparative Bar Chart")
        fig_bar = get_plotly_model_comparison(df_comp)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.markdown("#### Interactive Confusion Matrix Inspector")
        model_names = list(metrics_dict.keys())
        chosen_m = st.selectbox("Select Model to Inspect:", model_names, index=min(3, len(model_names) - 1))

        chosen_meta = metrics_dict[chosen_m]
        cm_data = np.array(chosen_meta["confusion_matrix"])
        c_names = chosen_meta.get("class_names", ["BENIGN", "ATTACK"])

        cm_norm = cm_data.astype(float) / (cm_data.sum(axis=1)[:, np.newaxis] + 1e-9)

        text_matrix = []
        for i in range(cm_data.shape[0]):
            row_txt = []
            for j in range(cm_data.shape[1]):
                row_txt.append(f"{cm_data[i, j]:,}<br>({cm_norm[i, j]*100:.1f}%)")
            text_matrix.append(row_txt)

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=c_names,
            y=c_names,
            text=text_matrix,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        ))
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8F9FA"),
            title=f"Confusion Matrix: {chosen_m.replace('_', ' ').title()}",
            xaxis_title="Predicted Label",
            yaxis_title="Ground Truth Label",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab3:
        st.markdown("#### Receiver Operating Characteristic (ROC) Comparison")
        roc_map = {}
        for m_name, m_data in metrics_dict.items():
            if m_data.get("roc_curve"):
                r_c = m_data["roc_curve"]
                roc_map[m_name.replace("_", " ").title()] = (
                    np.array(r_c["fpr"]),
                    np.array(r_c["tpr"]),
                    r_c["auc"],
                )

        if roc_map:
            fig_roc = get_plotly_roc_curves(roc_map)
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC curve data not available for evaluated models.")

    with tab4:
        st.markdown("#### Operational Tradeoffs: Detection Capability vs. Pipeline Latency")
        fig_scatter = px.scatter(
            df_comp,
            x="latency_ms_per_1k",
            y="macro_f1",
            text="model_name",
            size=[20] * len(df_comp),
            color="accuracy",
            color_continuous_scale="Viridis",
            title="Macro-F1 vs. Classification Latency (per 1,000 Flows)",
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8F9FA"),
            xaxis_title="Inference Latency (Milliseconds per 1,000 flows)",
            yaxis_title="Macro F1-Score",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Lower latency and higher Macro-F1 represent the optimal operating frontier.")

