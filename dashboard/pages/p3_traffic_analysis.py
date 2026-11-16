"""
Page 3: Network Traffic Analysis & Statistical Profiling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.visualization import SOC_PALETTE


def render_page(df: pd.DataFrame):
    st.markdown("## 🌐 Network Traffic Analysis & Flow Telemetry")
    st.markdown("Statistical profiling of flow durations, transport rates, packet distributions, and feature correlations.")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tab1, tab2, tab3 = st.tabs(["⏱️ Flow Rates & Durations", "🎯 Transport Ports & Protocols", "🔗 Feature Correlations"])

    with tab1:
        st.markdown("#### Flow Duration & Rate Distributions")
        c1, c2 = st.columns(2)

        with c1:
            if "Flow Duration" in df.columns:
                fig_dur = px.histogram(
                    df[df["Flow Duration"] > 0],
                    x="Flow Duration",
                    nbins=40,
                    log_x=True,
                    title="Flow Duration Distribution (Microseconds - Log Scale)",
                    color_discrete_sequence=[SOC_PALETTE["electric_blue"]],
                )
                fig_dur.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8F9FA"))
                st.plotly_chart(fig_dur, use_container_width=True)

        with c2:
            if "Flow Bytes/s" in df.columns:
                clean_bytes = df["Flow Bytes/s"].replace([np.inf, -np.inf], np.nan).dropna()
                clean_bytes = clean_bytes[clean_bytes > 0]
                fig_bytes = px.histogram(
                    clean_bytes,
                    nbins=40,
                    log_x=True,
                    title="Flow Bytes/Second Distribution (Log Scale)",
                    color_discrete_sequence=[SOC_PALETTE["cyan"]],
                )
                fig_bytes.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8F9FA"))
                st.plotly_chart(fig_bytes, use_container_width=True)

        st.caption("Observation: Network flows exhibit extreme right-skewness spanning multiple orders of magnitude.")

    with tab2:
        st.markdown("#### Service Ports & Traffic Concentration")
        if "Destination Port" in df.columns:
            port_counts = df["Destination Port"].value_counts().head(10).reset_index()
            port_counts.columns = ["Destination_Port", "Flow_Count"]
            port_counts["Port_Str"] = port_counts["Destination_Port"].astype(str)

            fig_ports = px.bar(
                port_counts,
                x="Port_Str",
                y="Flow_Count",
                color="Flow_Count",
                color_continuous_scale="Teal",
                title="Top 10 Most Active Destination Ports",
            )
            fig_ports.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8F9FA"))
            st.plotly_chart(fig_ports, use_container_width=True)
            st.caption("Key Ports: 80 (HTTP), 443 (HTTPS), 22 (SSH), 21 (FTP), 53 (DNS).")

    with tab3:
        st.markdown("#### Interactive Correlation Matrix")
        selected_features = st.multiselect(
            "Select Features for Correlation Analysis:",
            options=num_cols,
            default=num_cols[:12],
        )

        if len(selected_features) >= 2:
            sub_df = df[selected_features].replace([np.inf, -np.inf], np.nan).fillna(0)
            corr_mat = sub_df.corr().round(3)

            fig_corr = px.imshow(
                corr_mat,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Spearman / Pearson Linear Correlation",
            )
            fig_corr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8F9FA"))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Select at least 2 features to render the correlation heatmap.")

