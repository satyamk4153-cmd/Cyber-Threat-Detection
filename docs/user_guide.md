# Operational User Guide & Deployment Reference

## 1. System Requirements & Environment Setup
- **Operating Systems:** Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python Version:** Python 3.11 to 3.14 compatible
- **Hardware Minimums:** 4 Core CPU, 8 GB RAM (16 GB recommended for multi-GB raw dataset training)

### Step 1: Clone Repository & Create Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS (Bash)
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 2. Command-Line Interface (CLI) Guide
The system provides a unified entrypoint via `main.py`:

| Command | Action Performed |
| :--- | :--- |
| `python main.py` | Audits dataset status and displays CLI command options. |
| `python main.py --check-dataset` | Inspects `data/raw/` for the 8 official CIC-IDS2017 files. |
| `python main.py --generate-sample` | Generates a new verified software test fixture in `data/sample/`. |
| `python main.py --train` | Trains and evaluates all 6 binary models, saving artifacts to `models/`. |
| `python main.py --train --multiclass` | Trains and evaluates multiclass attack models. |
| `python main.py --anomaly` | Fits and evaluates Isolation Forest anomaly detection on benign flows. |
| `python main.py --explain` | Computes feature importance and attribution rankings. |
| `python main.py --predict <path.csv>` | Performs offline threat classification on an uploaded CSV capture. |
| `python main.py --dashboard` | Launches the interactive Streamlit SOC dashboard. |

---

## 3. Running Jupyter Notebooks
To run interactive step-by-step notebooks:
```bash
jupyter lab
# Or
jupyter notebook
```
Navigate to the `notebooks/` directory and execute sequentially:
- `01_Data_Understanding.ipynb`
- `02_Data_Cleaning_EDA.ipynb`
- `03_Exploratory_Data_Analysis.ipynb`
- `04_Feature_Engineering.ipynb`
- `05_Binary_Classification.ipynb`
- `06_Multiclass_Classification.ipynb`
- `07_Anomaly_Detection.ipynb`
- `08_Model_Comparison.ipynb`

---

## 4. Launching the Streamlit SOC Dashboard
Launch the web interface via:
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

### Navigating the 9 SOC Analytics Pages:
1. **🛡️ 1. Security Overview:** High-level executive KPI telemetry, attack ratios, and model status.
2. **📂 2. Dataset Explorer:** Record filtering, schema inspection, missing/infinite value counts.
3. **🌐 3. Network Traffic Analysis:** Flow duration, byte rate distributions, and correlation heatmap.
4. **🚨 4. Attack Intelligence:** Attack category descriptions and feature comparison boxplots.
5. **⚡ 5. ML Model Performance:** Benchmark tables, confusion matrices, ROC curves, and latency metrics.
6. **🔍 6. Anomaly Detection:** Isolation Forest score distributions, detection rates, and false alarms.
7. **🧠 7. Feature Importance:** Gini MDI, permutation importance, and causal disclaimers.
8. **🧪 8. CSV Prediction Lab:** Upload offline captures for batch classification and risk tiering.
9. **📘 9. About & Academic Project:** Research methodology, objectives, and literature citations.

---

## 5. Offline CSV Prediction Lab Workflow
1. Navigate to **Page 8: CSV Prediction Lab** on the dashboard.
2. Click **Browse files** and select any offline network flow capture in `.csv` format.
3. The platform validates file size, strips header whitespace, aligns columns with the trained model's feature set, and imputes any missing columns.
4. Review the **Threat Detection Summary**, **Risk Tier Breakdown**, and inspect individual classified flows.
5. Click **Download Threat Classification Report (CSV)** to export labeled results.

---

## 6. Troubleshooting & Diagnostics

### Q: Why does the dashboard show "Software Verification Mode Active"?
**Answer:** The system checks `data/raw/` for the official 8 CIC-IDS2017 CSV files. If they have not been placed there, it automatically runs in verified software test mode using `data/sample/cicids2017_sample.csv` so all features, tests, and interfaces can be demonstrated without crashing.

### Q: How do I run unit tests?
**Answer:** Execute:
```bash
python -m pytest -q
```
All 16 unit tests across loading, preprocessing, validation, models, and prediction will run and verify system integrity.

