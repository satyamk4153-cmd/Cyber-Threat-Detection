"""
Page 8: Offline CSV Threat Prediction Laboratory
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
from src.config import config
from src.predict import load_model_pipeline, predict_traffic_records
from src.data_validation import validate_csv_file, validate_dataframe_schema


def render_page(selected_model_name: str):
    st.markdown("## 🧪 Offline CSV Threat Prediction Laboratory")
    st.markdown("Ingest offline network traffic captures in CSV format for automated threat classification, confidence scoring, and risk tiering.")

    st.info(
        "🔒 **Defensive Cybersecurity Policy:** Uploaded files are evaluated strictly as offline tabular text. "
        "No network traffic is transmitted externally, and no uploaded files are ever executed."
    )

    # File Uploader
    uploaded_file = st.file_uploader(
        "Choose a network traffic CSV file:",
        type=["csv"],
        help="Upload an offline CICFlowMeter or standard network flow CSV capture.",
    )

    # Option to use sample records for immediate test
    use_sample = st.checkbox("Or test with 50 pre-validated sample records from test holdout", value=False)

    df_to_predict = None

    if uploaded_file is not None:
        file_check = validate_csv_file(uploaded_file)
        if not file_check["is_valid"]:
            st.error(f"❌ File Validation Error: {file_check['error']}")
            return

        try:
            df_to_predict = pd.read_csv(uploaded_file, low_memory=False)
            st.success(f"Loaded '{uploaded_file.name}' successfully ({len(df_to_predict):,} rows).")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

    elif use_sample:
        sample_path = config.sample_data_dir / "cicids2017_sample.csv"
        if sample_path.exists():
            full_sample = pd.read_csv(sample_path, low_memory=False)
            df_to_predict = full_sample.sample(n=min(50, len(full_sample)), random_state=42)
            st.info(f"Loaded 50 representative sample records.")

    if df_to_predict is not None:
        # Schema validation
        schema_status = validate_dataframe_schema(df_to_predict, allow_missing_target=True)
        if not schema_status["is_valid"]:
            st.warning(f"⚠️ Schema Notice: {schema_status.get('error')}. Missing columns will be imputed with zero/medians.")

        if st.button("🚀 Execute Threat Detection Pipeline", type="primary"):
            with st.spinner("Analyzing network flows with trained ML model..."):
                try:
                    pipeline = load_model_pipeline()
                    results_df, summary = predict_traffic_records(df_to_predict, pipeline)

                    st.markdown("### 📋 Threat Detection Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Flows Analyzed", f"{summary['total_records_analyzed']:,}")
                    c2.metric("Benign Flows", f"{summary['benign_records']:,}")
                    c3.metric("Attacks Detected", f"{summary['attack_records']:,}", f"{summary['threat_ratio_percentage']}%")
                    c4.metric("Active Model", summary["model_used"])

                    st.markdown("#### 🚨 Risk Breakdown")
                    st.write(summary["risk_breakdown"])

                    st.markdown("#### 🔍 Classified Traffic Observations")
                    cols_to_show = [
                        c for c in ["Destination Port", "Flow Duration", "Total Fwd Packets", "Packet Length Mean"]
                        if c in results_df.columns
                    ] + ["Predicted_Threat_Label", "Confidence_Score", "Risk_Tier"]

                    st.dataframe(results_df[cols_to_show].head(100), use_container_width=True)

                    # Export button
                    csv_export = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Threat Classification Report (CSV)",
                        data=csv_export,
                        file_name="cyber_threat_predictions.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Inference execution failed: {e}")

