"""
Page 7: Feature Importance, Explainability & Attribution
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.config import config
from src.predict import load_model_pipeline
from src.explainability import (
    extract_tree_feature_importance,
    calculate_permutation_importance,
    get_causality_disclaimer,
)
from src.preprocessing import prepare_train_test_split
from src.visualization import get_plotly_feature_importance


def render_page(df: pd.DataFrame):
    st.markdown("## 🧠 Model Explainability & Feature Importance")
    st.markdown("Attribution analysis revealing which network flow features exert the highest influence on threat classifications.")

    try:
        pipeline = load_model_pipeline()
    except Exception as e:
        st.warning(f"⚠️ Could not load trained pipeline: {e}")
        return

    model = pipeline["model"]
    feature_names = pipeline["features"]

    tab1, tab2, tab3 = st.tabs(["📊 Gini MDI Importance", "🔄 Permutation Importance", "🛡️ Causality vs Correlation"])

    with tab1:
        st.markdown("#### Mean Decrease in Impurity (Tree Gini Importance)")
        if hasattr(model, "feature_importances_"):
            imp_df = extract_tree_feature_importance(model, feature_names)

            top_n = st.slider("Select Number of Top Features to Display:", 5, 30, 15)

            imp_dict = dict(zip(imp_df["feature"], imp_df["importance"]))
            fig_imp = get_plotly_feature_importance(imp_dict, top_n=top_n)
            st.plotly_chart(fig_imp, use_container_width=True)

            st.dataframe(imp_df.head(top_n), use_container_width=True)
        else:
            st.info("The active model does not utilize tree impurity feature importances.")

    with tab2:
        st.markdown("#### Permutation Importance on Holdout Data")
        st.caption("Permutation importance breaks the association between a feature and the target on holdout test data, measuring drop in F1-score.")

        if st.button("Compute Permutation Importance (3 Repeats)"):
            with st.spinner("Computing permutation shuffles..."):
                X_tr, X_te, y_tr, y_te, _, _ = prepare_train_test_split(df, mode=pipeline.get("target_mode", "binary"))
                perm_df = calculate_permutation_importance(model, X_te, y_te, feature_names, n_repeats=3)

                fig_perm = px.bar(
                    perm_df.head(15),
                    x="importance_mean",
                    y="feature",
                    error_x="importance_std",
                    orientation="h",
                    color="importance_mean",
                    color_continuous_scale="Purples",
                    title="Top 15 Permutation Importance Features (Macro-F1 Drop)",
                )
                fig_perm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8F9FA"), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_perm, use_container_width=True)
                st.dataframe(perm_df.head(15), use_container_width=True)

    with tab3:
        st.markdown("#### Research Disclaimer: Importance vs. Causality")
        st.warning(get_causality_disclaimer())
        st.markdown(
            """
            **Key Insights from Cyber Traffic Attribution:**
            - **Flow Duration & Rates:** High forward packet rates and short inter-arrival times correlate heavily with automated flood scripts (DDoS, Hulk).
            - **Control Flags:** Skewed SYN-to-ACK ratios flag scanning reconnaissance (PortScan).
            - **Window Sizes:** Unusual TCP initial window bytes frequently reveal non-standard packet crafting tools (Scapy, Nmap, Hping).
            """
        )

