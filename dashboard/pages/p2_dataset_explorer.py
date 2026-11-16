"""
Page 2: Dataset Explorer & Schema Inspector
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import inspect_dataset, check_raw_dataset_status


def render_page(df: pd.DataFrame, dataset_meta: dict):
    st.markdown("## 📂 Dataset Explorer & Schema Inspector")
    st.markdown("Deep inspection of network flow schemas, data quality metrics, and missing/infinite value distributions.")

    summary = inspect_dataset(df)

    # Metrics summary row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{summary['total_rows']:,}")
    c2.metric("Total Features", f"{summary['total_columns']}")
    c3.metric("Missing Values", f"{summary['total_missing_values']:,}")
    c4.metric("Infinite Values", f"{summary['total_infinite_values']:,}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Data Preview & Filter",
        "📊 Data Quality Audit",
        "🏷️ Class Distribution",
        "📥 Acquisition & Download Guide",
    ])

    with tab1:
        st.markdown("#### Interactive Record Inspection")
        target_col = "Label" if "Label" in df.columns else df.columns[-1]

        # Filter by class
        all_classes = ["All"] + list(df[target_col].unique())
        selected_class = st.selectbox("Filter by Traffic Category:", all_classes)

        filtered_df = df if selected_class == "All" else df[df[target_col] == selected_class]

        # Search column selector
        col_subset = st.multiselect(
            "Select Features to Display:",
            options=list(df.columns),
            default=list(df.columns[:8]) + [target_col] if target_col in df.columns else list(df.columns[:8]),
        )

        st.dataframe(filtered_df[col_subset].head(100), use_container_width=True)
        st.caption(f"Displaying top 100 matching rows of {len(filtered_df):,} total filtered records.")

    with tab2:
        st.markdown("#### Forensic Data Quality Assessment")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### ⚠️ Infinite Value Occurrences")
            inf_dict = summary["columns_with_infinite"]
            if inf_dict:
                df_inf = pd.DataFrame(list(inf_dict.items()), columns=["Feature", "Inf_Count"])
                st.dataframe(df_inf, use_container_width=True)
                st.caption("Common CIC-IDS2017 artifact caused by zero flow durations. Replaced with NaN and imputed with medians.")
            else:
                st.success("No infinite values detected in current dataset.")

            st.markdown("##### 🔄 Duplicate Flows")
            st.write(f"**Duplicate Count:** {summary['duplicate_rows']:,} rows")

        with col_b:
            st.markdown("##### ❓ Missing Values (NaN) by Feature")
            missing_dict = summary["columns_with_missing"]
            if missing_dict:
                df_miss = pd.DataFrame(list(missing_dict.items()), columns=["Feature", "Missing_Count"])
                st.dataframe(df_miss, use_container_width=True)
            else:
                st.success("No missing values detected.")

            st.markdown("##### 💾 Memory Utilization")
            st.write(f"**Deep Memory Footprint:** {summary['memory_usage_mb']:.2f} MB")

    with tab3:
        st.markdown("#### Attack Category Frequencies")
        counts_df = pd.DataFrame(list(summary["label_counts"].items()), columns=["Category", "Frequency"])
        counts_df["Percentage"] = (counts_df["Frequency"] / len(df) * 100).round(2)
        st.dataframe(counts_df, use_container_width=True)

    with tab4:
        st.markdown("#### Official CIC-IDS2017 Acquisition Protocol")
        st.markdown(
            """
            To run experiments on the full multi-gigabyte dataset:
            1. Visit the Canadian Institute for Cybersecurity official portal:  
               [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
            2. Download **GeneratedLabelledFlows.zip**.
            3. Extract the 8 CSV files into:
               ```
               Cyber_Threat_Detection/data/raw/
               ```
            4. Reload the dashboard or restart the CLI. The system will automatically detect the raw files.
            """
        )

