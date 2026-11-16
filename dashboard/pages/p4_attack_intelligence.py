"""
Page 4: Attack Intelligence & Threat Signatures
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.visualization import SOC_PALETTE


def render_page(df: pd.DataFrame):
    st.markdown("## 🚨 Attack Intelligence & Threat Signatures")
    st.markdown("Forensic comparative analysis between benign traffic profiles and specific cyberattack categories.")

    target_col = "Label" if "Label" in df.columns else df.columns[-1]

    tab1, tab2, tab3 = st.tabs(["📊 Threat Profiles", "🔬 Feature Comparison", "⚖️ Imbalance Mitigation"])

    with tab1:
        st.markdown("#### Attack Category Profiles & Signatures")
        attack_descriptions = {
            "BENIGN": "Normal non-malicious user and background service communications.",
            "DoS Hulk": "High-volume HTTP request flood designed to exhaust web application thread pools.",
            "PortScan": "Reconnaissance scan probing multiple destination ports to discover active services.",
            "DDoS": "Distributed Denial of Service traffic overwhelming ingress firewall throughput.",
            "DoS GoldenEye": "HTTP keep-alive and cache-control starvation attack on web services.",
            "FTP-Patator": "Automated brute-force password guessing targeting Port 21 (FTP).",
            "SSH-Patator": "Automated dictionary attack targeting Port 22 (SSH).",
            "DoS slowloris": "Slow-rate HTTP header starvation holding socket connections open indefinitely.",
            "Bot": "ARES Botnet Command & Control (C2) beaconing and staging traffic.",
            "Web Attack - Brute Force": "HTTP form login brute force targeting administrative dashboards.",
        }

        desc_rows = []
        for cat, desc in attack_descriptions.items():
            cnt = int((df[target_col] == cat).sum()) if cat in df[target_col].values else 0
            desc_rows.append({"Category": cat, "Observed_Flows": cnt, "Attack Vector Description": desc})

        st.dataframe(pd.DataFrame(desc_rows), use_container_width=True)

    with tab2:
        st.markdown("#### Cross-Category Feature Behavior")
        compare_features = [
            c for c in ["Flow Duration", "Total Fwd Packets", "Packet Length Mean", "Flow Bytes/s", "Fwd Packets/s"]
            if c in df.columns
        ]

        if compare_features:
            selected_feat = st.selectbox("Select Feature to Compare Across Attack Categories:", compare_features)

            sub = df[[target_col, selected_feat]].copy()
            sub[selected_feat] = pd.to_numeric(sub[selected_feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

            fig_box = px.box(
                sub,
                x=target_col,
                y=selected_feat,
                color=target_col,
                points=False,
                title=f"Distribution of '{selected_feat}' Grouped by Traffic Category",
            )
            fig_box.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8F9FA"), showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        st.markdown("#### The Class Imbalance Challenge in Cybersecurity")
        st.info(
            """
            **Why Accuracy is a Dangerous Metric in Cyber Defense:**
            In production networks, malicious traffic accounts for less than 1% of total flow volume.
            A naive 'dumb' classifier predicting 'BENIGN' for 100% of flows achieves 99% accuracy while missing
            every single security breach.

            **Defensive Mitigations Applied:**
            1. **Macro-F1 Optimization:** Evaluates unweighted arithmetic mean of per-class F1 scores.
            2. **Cost-Sensitive Learning:** Dynamic inverse class weighting penalizes missed minority attacks.
            3. **Unsupervised Outlier Baselines:** Isolation Forest identifies anomalies without relying on balanced classes.
            """
        )

