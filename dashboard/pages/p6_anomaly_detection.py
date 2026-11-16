"""
Page 6: Unsupervised Anomaly Detection (Isolation Forest)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import config
from src.anomaly_detection import NetworkAnomalyDetector
from src.preprocessing import prepare_train_test_split
from src.visualization import SOC_PALETTE


def render_page(df: pd.DataFrame):
    st.markdown("## 🔍 Unsupervised Anomaly Detection (Isolation Forest)")
    st.markdown("Outlier identification and zero-day behavioral deviation analysis trained strictly on benign baseline communications.")

    target_col = "Label" if "Label" in df.columns else df.columns[-1]

    # Check if saved anomaly detector exists
    model_path = config.saved_models_dir / "isolation_forest.joblib"
    if not model_path.exists():
        st.warning("⚠️ Trained Isolation Forest model artifact not found. Please train models via CLI or main script.")
        return

    detector = NetworkAnomalyDetector.load(model_path)

    # Perform evaluation on current dataset sample
    X_tr, X_te, y_tr, y_te, prep, _ = prepare_train_test_split(df, mode="binary")
    eval_res = detector.evaluate_against_ground_truth(X_te, y_te)
    preds, scores = detector.predict_anomalies(X_te)

    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anomalies Flagged", f"{eval_res['anomalies_detected']:,}", f"{eval_res['anomaly_percentage']}% of flows")
    c2.metric("Detection Rate (Recall)", f"{eval_res['detection_rate_recall']*100:.1f}%", "True attacks caught")
    c3.metric("False Alarm Rate (FAR)", f"{eval_res['false_alarm_rate_fpr']*100:.1f}%", "Benign flows flagged")
    c4.metric("Empirical Threshold", f"{eval_res['decision_threshold']:.4f}", "Isolation boundary")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📉 Anomaly Score Distribution", "🎯 Diagnostic Matrix", "📘 Behavioral Methodology"])

    with tab1:
        st.markdown("#### Anomaly Score Distribution (Holdout Traffic)")
        df_scores = pd.DataFrame({"Anomaly_Score": scores, "True_Label": np.where(y_te == 1, "Attack", "Benign")})

        fig_score = px.histogram(
            df_scores,
            x="Anomaly_Score",
            color="True_Label",
            nbins=50,
            barmode="overlay",
            color_discrete_map={"Attack": SOC_PALETTE["alert_red"], "Benign": SOC_PALETTE["cyan"]},
            title="Isolation Forest Decision Function Score (Negative = More Anomalous)",
        )
        fig_score.add_vline(x=eval_res['decision_threshold'], line_dash="dash", line_color="#FFBE0B", annotation_text="Threshold")
        fig_score.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8F9FA"))
        st.plotly_chart(fig_score, use_container_width=True)
        st.caption("Lower decision scores indicate rapid tree isolation (characteristic of rare, unusual flow topologies).")

    with tab2:
        st.markdown("#### Anomaly Confusion Breakdown")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.write(f"**True Positives (Attacks Flagged):** {eval_res['true_positives_attacks_flagged']:,}")
            st.write(f"**False Positives (Benign Alarms):** {eval_res['false_positives_benign_alarms']:,}")

        with col_m2:
            st.write(f"**True Negatives (Benign Passed):** {eval_res['true_negatives_benign_passed']:,}")
            st.write(f"**False Negatives (Attacks Missed):** {eval_res['false_negatives_attacks_missed']:,}")

    with tab3:
        st.markdown("#### The Role of Anomaly Detection in Cyber Defense")
        st.info(
            """
            **Supervised Learning vs. Unsupervised Anomaly Detection:**
            - **Supervised Models (e.g. Random Forest):** Learn precise boundaries for specific known attack signatures
              (e.g., DoS Hulk HTTP headers). However, novel or mutated zero-day attacks can bypass these boundaries.
            - **Isolation Forest:** Builds tree ensembles purely on normal benign communications.
              Any flow that isolates quickly in random decision space is flagged as an anomaly.

            **Critical Defensive Principle:**
            An anomaly is a statistical outlier, NOT proof of malice. Large video downloads, software updates,
            or network misconfigurations frequently cause benign anomaly alarms.
            """
        )

