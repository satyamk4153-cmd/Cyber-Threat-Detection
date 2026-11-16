"""
Page 9: About Project, Academic Methodology & Defensive Governance
"""

import streamlit as st


def render_page():
    st.markdown("## 📘 About the Project & Research Methodology")
    st.markdown(
        """
        **Intelligent Cyber Threat Detection & Network Traffic Analysis Using Python and Machine Learning**  
        *An Academic Data Science and Defensive Cybersecurity Research Platform.*
        """
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Objectives & Scope",
        "⚙️ System Architecture",
        "⚖️ Ethics & Defensive Policy",
        "📚 Academic References",
    ])

    with tab1:
        st.markdown(
            """
            ### Project Objectives
            1. **Understand Normal vs Attack Behaviors:** Quantify bidirectional flow statistics that distinguish standard web browsing from malicious exploitation.
            2. **Rigorous Machine Learning Benchmarks:** Compare linear models (Logistic Regression), tree-based interpretable models (Decision Trees), ensemble models (Random Forest, Gradient Boosting, XGBoost), and unsupervised anomaly detectors (Isolation Forest).
            3. **Address Real-World Class Imbalance:** Implement cost-sensitive learning and prioritize **Macro-F1** over misleading overall accuracy.
            4. **Explainability & Attribution:** Isolate the top discriminative features via Gini impurity and holdout permutation importance.
            5. **Operational Triaging:** Build an interactive SOC-grade dashboard for cybersecurity analysts.
            """
        )

    with tab2:
        st.markdown(
            """
            ### Modular System Architecture
            - **Data Ingestion (`src/data_loader.py`):** Handles multi-CSV loading, memory downcasting, and whitespace cleanup.
            - **Pre-processing (`src/preprocessing.py`):** Median imputation, constant column elimination, and leakage prevention.
            - **Domain Engineering (`src/feature_engineering.py`):** Forward/backward volume asymmetry, TCP control flags, and packet dispersion metrics.
            - **Model Suite (`src/train.py`):** Trains and persists supervised classifiers with full metadata tracking.
            - **Evaluation Diagnostics (`src/evaluate.py`):** Computes ROC-AUC, PR-AUC, confusion matrices, FPR, and classification latency.
            - **Anomaly Detection (`src/anomaly_detection.py`):** Unsupervised Isolation Forest for out-of-distribution traffic.
            - **Prediction Lab (`src/predict.py`):** End-to-end safe CSV batch inference with confidence scores and risk tiers.
            """
        )

    with tab3:
        st.markdown(
            """
            ### Defensive Cybersecurity Principles
            - **No Active Exploitation:** The system operates exclusively on offline benchmark captures (CIC-IDS2017) or user-supplied offline flow logs.
            - **No External Probing:** No network scanners, port knockers, packet injectors, or exploit payloads are executed.
            - **No Dynamic Code Execution:** All uploaded files are parsed strictly as tabular text.
            - **Zero Data Leakage:** All preprocessing scalers and encoders are fit strictly on training partitions.
            """
        )

    with tab4:
        st.markdown(
            """
            ### Academic Literature & Primary References
            1. **Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018).**  
               *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.*  
               Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal.
            2. **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).**  
               *Isolation Forest.*  
               Eighth IEEE International Conference on Data Mining (ICDM), Pisa, Italy.
            3. **Breiman, L. (2001).**  
               *Random Forests.*  
               Machine Learning, 45(1), 5-32.
            4. **Pedregosa, F. et al. (2011).**  
               *Scikit-learn: Machine Learning in Python.*  
               Journal of Machine Learning Research, 12, 2825-2830.
            """
        )

