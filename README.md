# 🛡️ Intelligent Cyber Threat Detection & Network Traffic Analysis

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS2017-purple.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![Test Suite](https://img.shields.io/badge/Tests-16%20Passed-brightgreen.svg)](tests/)

An advanced, modular, and defensible Machine Learning and Network Traffic Analysis platform designed for intelligent cyber threat detection. Developed for **NPTEL Python for Data Science**, **B.Tech CSE Capstone**, and professional cybersecurity engineering portfolios.

---

## 📌 Executive Summary & Project Vision
Legacy signature-based network intrusion detection systems struggle against encrypted traffic, polymorphism, and zero-day exploits. This project provides an end-to-end data science and machine learning research platform that analyzes bidirectional network flow records from the **CIC-IDS2017** benchmark dataset to identify malicious network activity.

### The platform addresses 8 core cybersecurity research questions:
1. **Normal vs. Hostile Profiles:** What statistical flow signatures distinguish legitimate user activity from attacks?
2. **Attack Vector Topologies:** How do DoS, DDoS, PortScan, Patator Brute Force, and Botnets differ in transport space?
3. **Severe Class Imbalance:** Why does standard accuracy fail in cybersecurity, and how does **Macro-F1** and cost-sensitive learning resolve it?
4. **Feature Attribution:** Which transport parameters (inter-arrival times, packet variance, TCP flags) are most informative?
5. **Multi-Algorithm Benchmarking:** How do linear baselines, decision trees, tree bagging ensembles, gradient boosting, and XGBoost compare?
6. **Error Analysis:** What are the operational impacts of **False Positive Rate (FPR)** and **False Negative Rate (FNR)** in SOC environments?
7. **Zero-Day Anomaly Detection:** How can **Isolation Forest** detect out-of-distribution traffic without attack signatures?
8. **Operational Deployment:** How can trained pipelines be served in an interactive, dark-themed SOC interface with batch CSV inference?

---

## ⚙️ Core Technology Stack
- **Language:** Python 3.11 - 3.14
- **Data Manipulation & Mathematics:** NumPy, Pandas, SciPy
- **Machine Learning & Boosting:** Scikit-learn, XGBoost, Joblib
- **Explainability:** Tree Gini Impurity (MDI), Holdout Permutation Importance, SHAP
- **Interactive Visualization & Dashboard:** Streamlit, Plotly, Matplotlib, Seaborn
- **Configuration & Persistence:** PyYAML, JSON metadata, Joblib pipelines
- **Quality Assurance & Testing:** Pytest (16 comprehensive unit tests), compileall

---

## 🏛️ System Architecture
```
Cyber_Threat_Detection/
│
├── data/
│   ├── raw/                  # Multi-gigabyte official CIC-IDS2017 CSV files
│   ├── processed/            # Intermediate preprocessed datasets
│   └── sample/               # Verified schema software testing fixture
│
├── notebooks/                # 8 Complete, Sequential Research Notebooks
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Cleaning_EDA.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Binary_Classification.ipynb
│   ├── 06_Multiclass_Classification.ipynb
│   ├── 07_Anomaly_Detection.ipynb
│   └── 08_Model_Comparison.ipynb
│
├── src/                      # Modular, Reusable Python Package
│   ├── __init__.py
│   ├── config.py             # Singleton configuration manager (config.yaml)
│   ├── data_loader.py        # Chunked multi-file loader & dataset auditor
│   ├── data_validation.py    # Schema checks & upload file security
│   ├── preprocessing.py      # Anti-leakage preprocessor & imputer
│   ├── feature_engineering.py# Domain asymmetry ratios & collinearity filters
│   ├── visualization.py      # Publication Seaborn figures & Plotly charts
│   ├── train.py              # Supervised model builders & training suite
│   ├── evaluate.py           # Diagnostic metrics (FPR, FNR, ROC, Latency)
│   ├── anomaly_detection.py  # Isolation Forest on benign baseline
│   ├── explainability.py     # Attribution, Gini MDI & Permutation importance
│   └── predict.py            # Offline batch CSV inference with risk tiers
│
├── models/
│   ├── saved_models/         # Serialized .joblib pipeline packages
│   └── model_metadata/       # Full JSON audit trails per model
│
├── dashboard/                # Modern SOC Streamlit Application
│   ├── app.py                # Main web entrypoint with custom dark SOC CSS
│   ├── components/           # Reusable KPI cards & sidebar telemetry
│   └── pages/                # 9 Dedicated analytics and prediction pages
│
├── reports/
│   ├── figures/              # High-resolution publication plots (.png)
│   ├── results/              # Comparative benchmark CSVs & JSON metrics
│   └── project_report.md     # 25-Section Formal Academic Report
│
├── tests/                    # 16 Automated Pytest Unit Tests
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_validation.py
│   ├── test_models.py
│   └── test_prediction.py
│
├── docs/                     # Comprehensive Engineering Guides
│   ├── methodology.md        # Research methodology & leakage protocol
│   ├── architecture.md       # Architectural blueprints & Mermaid diagrams
│   ├── dataset_guide.md      # Official UNB acquisition instructions
│   └── user_guide.md         # Deployment, CLI, & troubleshooting guide
│
├── requirements.txt          # Python dependency specifications
├── config.yaml               # Centralized hyperparameter & path settings
├── main.py                   # Unified Command-Line Interface (CLI)
└── .gitignore                # Git exclusion specifications
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Virtual Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/Cyber_Threat_Detection.git
cd Cyber_Threat_Detection

# Create virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Create virtual environment (Linux / macOS)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Dataset Status
```bash
python main.py --check-dataset
```
*Note: If the raw CIC-IDS2017 files are not yet in `data/raw/`, the system automatically enters verified software testing mode using `data/sample/cicids2017_sample.csv` so all modules, tests, and dashboards run immediately.*

### 3. Run Automated Unit Tests
```bash
python -m pytest -q
```
*Expected Output: 16 passed in ~3.30s.*

---

## 🖥️ Command-Line Interface (CLI) Reference

```bash
# Train all binary classification models
python main.py --train

# Train multiclass attack category models
python main.py --train --multiclass

# Train and evaluate Isolation Forest anomaly detection
python main.py --anomaly

# Display top influential feature rankings
python main.py --explain

# Run offline batch threat inference on any network flow CSV
python main.py --predict data/sample/cicids2017_sample.csv

# Launch interactive Streamlit SOC dashboard
python main.py --dashboard
```

---

## 📊 Experimental Results & Leaderboard

### Supervised Binary Classification (Holdout Test Evaluation)
*Evaluated on holdout test partition with strict zero-leakage preprocessing:*

| Algorithm | Overall Accuracy | Precision | Recall | Macro-F1 | False Positive Rate (FPR) | False Negative Rate (FNR) | Latency (ms / 1k flows) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 53.76% | 0.5262 | 0.4908 | **0.5074** | 39.30% | 62.94% | 0.81 ms |
| **Logistic Regression** | 51.68% | 0.5103 | 0.5043 | **0.5050** | 46.55% | 52.59% | 0.79 ms |
| **Stratified Dummy (Base)** | 58.88% | 0.4912 | 0.5060 | **0.4939** | 29.33% | 69.48% | 0.42 ms |
| **Random Forest** | 65.92% | 0.5323 | 0.5357 | **0.4633** | 9.40% | 93.46% | 12.45 ms |
| **Gradient Boosting** | 68.80% | 0.4856 | 0.5029 | **0.4578** | 4.87% | 94.55% | 2.15 ms |
| **XGBoost** | **68.48%** | 0.4402 | 0.4927 | **0.4404** | **4.19%** | 97.28% | 3.40 ms |

### Multiclass Classification Performance
| Algorithm | Accuracy | Macro-F1 | Weighted-F1 | Training Duration |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 40.96% | **0.1012** | 0.3845 | 0.42 s |
| **Gradient Boosting** | 68.08% | **0.0847** | 0.5812 | 1.85 s |
| **XGBoost** | **70.56%** | **0.0827** | **0.6014** | 0.95 s |
| **Decision Tree** | 15.84% | **0.0642** | 0.1830 | 0.06 s |

### Unsupervised Isolation Forest Anomaly Detection
- **Fitted Baseline:** 2,647 Benign Flows
- **Holdout Test Flows:** 1,250 Flows
- **Anomalies Detected:** 178 Flows (14.24%)
- **Attack Detection Rate (Recall):** 13.90%
- **False Alarm Rate (FAR / FPR):** 14.38%
- **Empirical Threshold:** -0.0000

---

## 🛡️ Streamlit SOC Analytics Interface
Launch via:
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

### 9 Dedicated SOC Analytics Pages:
1. **🛡️ 1. Security Overview:** Real-time KPI telemetry cards, benign vs threat ratio donut chart, and attack distribution.
2. **📂 2. Dataset Explorer:** Interactive row filtering, deep memory usage, missing and infinite value audit.
3. **🌐 3. Network Traffic Analysis:** Flow duration histograms, flow byte rates, and interactive correlation heatmaps.
4. **🚨 4. Attack Intelligence:** Attack category descriptions and feature comparison boxplots.
5. **⚡ 5. ML Model Performance:** Benchmark leaderboards, normalized confusion matrices, and ROC curves.
6. **🔍 6. Anomaly Detection:** Isolation Forest decision score histograms, detection rates, and false alarm breakdowns.
7. **🧠 7. Feature Importance:** Tree Gini MDI, holdout permutation importance, and causal disclaimers.
8. **🧪 8. CSV Prediction Lab:** Upload offline CSV flow captures for batch inference, confidence scoring, and risk tiering.
9. **📘 9. About & Academic Project:** Research methodology, system architecture, and academic citations.

---

## 🔒 Defensive Cybersecurity & Ethical Policy
- **Offline Analytics Only:** All operations are conducted strictly on offline PCAP-derived flow captures.
- **No Unauthorized Operations:** The system does NOT perform port scanning, network probing, packet injection, or exploit delivery.
- **Safe Input Handling:** File uploads in the CSV Prediction Lab are parsed strictly as tabular data without shell evaluation.

---

## 📚 References & Literature Citations
1. **Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018).** *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal.
2. **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).** *Isolation Forest.* Eighth IEEE International Conference on Data Mining (ICDM), Pisa, Italy.
3. **Breiman, L. (2001).** *Random Forests.* Machine Learning, 45(1), 5-32.
4. **Chen, T., & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.* ACM SIGKDD, 785-794.
5. **Pedregosa, F. et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.

---

## 📄 License
This project is released under the **MIT License**. See `LICENSE` for details.

# Cyber-Threat-Detection
