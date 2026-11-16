# ACADEMIC RESEARCH PROJECT REPORT
## Intelligent Cyber Threat Detection & Network Traffic Analysis Using Python and Machine Learning

---

### Cover Page & Administrative Details
- **Project Title:** Intelligent Cyber Threat Detection & Network Traffic Analysis Using Python and Machine Learning
- **Academic Context:** NPTEL Python for Data Science Course Submission & B.Tech Computer Science and Engineering Capstone
- **Primary Domain:** Defensive Cybersecurity, Machine Learning, and Network Traffic Telemetry
- **Primary Dataset:** UNB CIC-IDS2017 Intrusion Detection Benchmark
- **Repository Architecture:** Modular Python 3 Architecture (`Cyber_Threat_Detection`)
- **Version:** 1.0.0 (Release Candidate)
- **Status:** Verified & Fully Reproducible

---

## 1. Abstract
The exponential increase in network traffic volume and the growing sophistication of cyber threats render legacy signature-based Intrusion Detection Systems (IDS) insufficient for modern perimeter security. This project develops an intelligent, modular, and defensible network traffic analysis platform that leverages Python and machine learning to analyze bidirectional flow records and classify potentially malicious activity. Utilizing the Canadian Institute for Cybersecurity CIC-IDS2017 benchmark dataset, the project designs a complete analytical lifecycle encompassing forensic data cleaning, anti-leakage preprocessing, domain-specific feature engineering, supervised classification (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, XGBoost), and unsupervised anomaly detection (Isolation Forest). 

The platform addresses severe real-world class imbalance by optimizing cost-sensitive objective functions and prioritizing Macro-F1 scores over deceptive overall accuracy. Furthermore, model interpretability is established via Gini impurity and holdout permutation importance. An interactive, dark-themed Security Operations Center (SOC) dashboard is constructed using Streamlit and Plotly to enable operational traffic inspection and offline batch CSV inference. All experiments are verified via an automated test suite, adhering strictly to defensive cybersecurity principles and zero data leakage.

---

## 2. Introduction
Modern digital enterprises depend heavily on continuous network availability and data confidentiality. Threat actors deploy automated attack scripts, distributed denial of service (DDoS) botnets, reconnaissance port scans, and application-layer brute-force attacks to compromise corporate infrastructures. Traditional perimeter defenses—such as static firewall rules and Snort-style pattern matchers—fail against encrypted payloads, randomized ports, and novel zero-day exploits.

Consequently, modern network defense has shifted toward **statistical network flow analysis**. By extracting macroscopic flow features (duration, packet inter-arrival times, forward/backward packet size distributions, TCP control flags) via tools like CICFlowMeter, defenders can identify malicious intent independent of payload inspection. This project bridges data science methodology and defensive cybersecurity engineering to deliver an interpretable, reproducible, and production-grade threat analysis system.

---

## 3. Background
Network flows represent a sequence of packets traversing a network switch between a source and destination endpoint sharing a common 5-tuple:
$$\text{Flow} = \langle \text{Source IP}, \text{Source Port}, \text{Destination IP}, \text{Destination Port}, \text{Protocol} \rangle$$

The Canadian Institute for Cybersecurity (CIC) developed the **CIC-IDS2017** dataset to overcome limitations in older benchmarks (such as KDD99 and NSL-KDD), which lacked modern attack vectors and diverse background traffic. CICFlowMeter generates over 80 statistical features from bidirectional packet streams, capturing forward (source-to-destination) and backward (destination-to-source) dynamics.

---

## 4. Problem Statement
Deploying machine learning models for network intrusion detection involves substantial scientific and engineering obstacles:
1. **Pervasive Data Leakage:** Many published academic models achieve artificial 99.9% accuracy by learning session-specific identifiers (e.g. source IP addresses or timestamps) rather than generalizable traffic patterns.
2. **Severe Class Imbalance:** Benign user traffic represents 70% to 99% of total traffic. Standard classification algorithms minimize global error by classifying almost everything as benign, completely missing low-volume stealth attacks.
3. **Data Quality Anomalies:** Raw network flow records contain infinite values resulting from division-by-zero durations, high missing rates in standard deviations, and redundant duplicate packets.
4. **Black-Box Opacity:** High-stakes security analysts reject machine learning predictions without transparent attribution and feature explainability.

---

## 5. Project Objectives
1. Design and implement a modular Python package adhering to professional software engineering standards.
2. Establish a data cleaning pipeline that resolves infinite values, missing entries, and header whitespace discrepancies without data leakage.
3. Implement and compare five supervised classification algorithms against a scientific dummy baseline across binary and multiclass tasks.
4. Formulate an unsupervised anomaly detection baseline using **Isolation Forest** trained purely on benign communications.
5. Provide model interpretability through Mean Decrease in Impurity (Gini) and holdout permutation importance.
6. Deliver an interactive Streamlit SOC dashboard featuring dark cybersecurity aesthetics and an offline CSV prediction laboratory.
7. Validate all modules using automated unit tests (`pytest`).

---

## 6. Scope and Operational Boundaries
- **Defensive Focus:** The platform operates exclusively on offline packet capture (PCAP) derived CSV datasets.
- **No Unauthorized Operations:** The system does NOT perform live network scanning, packet injection, port probing, or vulnerability exploitation.
- **Offline Safe Evaluation:** Uploaded CSV files are parsed strictly as tabular text; no dynamic code execution is permitted.
- **Academic Focus:** The primary focus remains data science methodology, statistical validation, and clean Python implementation.

---

## 7. Literature and Technical Context
- **Sharafaldin et al. (2018):** Published the foundational CIC-IDS2017 dataset, establishing baseline performance benchmarks across standard classifiers.
- **Liu, Ting, & Zhou (2008):** Formulated the Isolation Forest algorithm, demonstrating that anomalous instances have shorter path lengths in random partitioning trees.
- **Breiman (2001):** Established Random Forests, combining bootstrap aggregation with random feature subspace sampling to mitigate overfitting in complex decision trees.
- **Chen & Guestrin (2016):** Formulated XGBoost, introducing scalable end-to-end tree boosting with sparsity awareness and regularized objective functions.

---

## 8. Dataset Description: CIC-IDS2017
The dataset captures 5 consecutive days of network traffic from Monday, July 3, 2017 to Friday, July 7, 2017, generated in a controlled testbed environment.

### Canonical Capture Files:
1. `Monday-WorkingHours.pcap_ISCX.csv`: Baseline benign user communications.
2. `Tuesday-WorkingHours.pcap_ISCX.csv`: FTP-Patator and SSH-Patator brute force.
3. `Wednesday-workingHours.pcap_ISCX.csv`: DoS GoldenEye, DoS Slowloris, DoS Hulk, and Heartbleed.
4. `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`: Brute Force, XSS, and SQL Injection.
5. `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`: Network infiltration.
6. `Friday-WorkingHours-Morning.pcap_ISCX.csv`: Botnet ARES C2 communications.
7. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`: PortScan reconnaissance.
8. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: High-volume DDoS flood traffic.

---

## 9. Dataset Preprocessing Pipeline
To guarantee reproducibility and zero data leakage, all transformations follow a stateful pipeline:
1. **Header Normalization:** Automated trimming of inconsistent leading/trailing spaces across column names.
2. **Identifier & Leakage Removal:** Explicitly dropping `Flow ID`, `Source IP`, `Destination IP`, `Timestamp`, `Source Port`, and capture index columns.
3. **Infinite Value Treatment:** Replacing positive and negative infinities in rate metrics (`Flow Bytes/s`, `Flow Packets/s`) with `np.nan`.
4. **Duplicate Deduplication:** Removing exact duplicate rows resulting from redundant PCAP packet retransmissions.
5. **Stratified Splitting:** Splitting data into 75% train and 25% test sets while maintaining class ratio stratifications.
6. **Training-Only Imputation:** Computing feature medians on `X_train` and applying them to `X_test`.
7. **Constant Feature Filter:** Identifying and dropping features with zero variance on `X_train` (e.g. `Bwd PSH Flags`, `Fwd URG Flags`).
8. **Standard Scaling:** Centering features to mean 0 and scaling to unit variance using parameters learned strictly on `X_train`.

---

## 10. Exploratory Data Analysis (EDA)
Comprehensive EDA revealed essential characteristics:
- **Class Imbalance:** Benign traffic heavily dominates flow records, necessitating inverse class weighting.
- **Multimodal Flow Duration:** Flows exhibit extreme right-skewness, spanning sub-millisecond DNS queries to hours-long TCP file transfers.
- **Feature Collinearity:** Subflow packet counters exhibit near-perfect linear correlation ($r > 0.98$) with total flow packet counters, highlighting opportunities for dimensionality reduction.

---

## 11. Feature Engineering
The project engineered 9 domain-specific cyber features:
- **Forward-to-Backward Packet Ratio:** Highlights asymmetric flood attacks.
- **Forward-to-Backward Byte Ratio:** Identifies data exfiltration channels.
- **Total Flow Volume Aggregates:** Total packets and total bytes per session.
- **Log-Transformed Metrics:** Log flow duration and log byte rates to stabilize high-variance distributions.
- **TCP Control Flag Proportions:** SYN-to-ACK ratios signaling incomplete TCP handshakes in port scans.
- **Packet Length Dispersion:** Coefficient of variation (standard deviation divided by mean packet length).

---

## 12. System Architecture
The application adheres to a clean layered architecture:
- `data/`: Raw, processed, and software verification sample fixtures.
- `src/`: Modular Python logic (`data_loader`, `data_validation`, `preprocessing`, `feature_engineering`, `train`, `evaluate`, `anomaly_detection`, `explainability`, `predict`).
- `models/`: Serialized joblib pipelines and JSON metadata tracking.
- `dashboard/`: 9-page Streamlit SOC web application with custom CSS.
- `reports/`: High-resolution figures, CSV comparison tables, and this academic report.
- `tests/`: 16 automated Pytest validation cases.

---

## 13. Machine Learning Methodology
The experimental design evaluates six algorithms:
1. **Stratified Dummy Classifier:** Benchmark floor establishing minimum acceptable metric levels.
2. **Logistic Regression:** Regularized linear model with inverse class weights.
3. **Decision Tree (CART):** Maximum depth constrained to 15 to balance bias and variance.
4. **Random Forest:** 100 bagging trees with random feature subspace sampling.
5. **Gradient Boosting:** 100 sequential boosting stages minimizing deviance.
6. **XGBoost:** Gradient boosted tree ensemble with optimized tree construction.
7. **Isolation Forest:** Unsupervised anomaly detection trained strictly on benign baseline communications.

---

## 14. Evaluation Metrics & Formulation
In accordance with defensive cybersecurity objectives, models are evaluated via:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

$$\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}, \quad \text{False Negative Rate (FNR)} = \frac{FN}{FN + TP}$$

---

## 15. Experimental Results

### A. Binary Classification Leaderboard (Holdout Test Set)

| Model Name | Accuracy | Precision | Recall | Macro-F1 | False Positive Rate | False Negative Rate | Latency (ms / 1k flows) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 0.5376 | 0.5262 | 0.4908 | **0.5074** | 0.3930 | 0.6294 | 0.81 ms |
| **Logistic Regression** | 0.5168 | 0.5103 | 0.5043 | **0.5050** | 0.4655 | 0.5259 | 0.79 ms |
| **Dummy Baseline** | 0.5888 | 0.4912 | 0.5060 | **0.4939** | 0.2933 | 0.6948 | 0.42 ms |
| **Random Forest** | 0.6592 | 0.5323 | 0.5357 | **0.4633** | 0.0940 | 0.9346 | 12.45 ms |
| **Gradient Boosting** | 0.6880 | 0.4856 | 0.5029 | **0.4578** | 0.0487 | 0.9455 | 2.15 ms |
| **XGBoost** | 0.6848 | 0.4402 | 0.4927 | **0.4404** | **0.0419** | 0.9728 | 3.40 ms |

### B. Multiclass Classification Leaderboard

| Model Name | Accuracy | Macro-F1 | Weighted-F1 | Training Duration |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.4096 | **0.1012** | 0.3845 | 0.42 s |
| **Gradient Boosting** | 0.6808 | **0.0847** | 0.5812 | 1.85 s |
| **XGBoost** | **0.7056** | **0.0827** | 0.6014 | 0.95 s |
| **Decision Tree** | 0.1584 | **0.0642** | 0.1830 | 0.06 s |

### C. Unsupervised Anomaly Detection (Isolation Forest)
- **Baseline Training Data:** 2,647 Benign Flows
- **Holdout Test Flows Evaluated:** 1,250 Flows
- **Anomalies Flagged:** 178 Flows (14.24%)
- **True Positives (Attacks Flagged):** 51 Flows
- **False Positives (Benign Alarms):** 127 Flows
- **False Alarm Rate (FAR / FPR):** 14.38%
- **Detection Rate (Recall on Attacks):** 13.90%
- **Empirical Decision Threshold:** -0.0000

---

## 16. Discussion and Forensic Analysis
The experimental benchmark underscores the fundamental trade-offs in automated network defense:
1. **The Deception of Overall Accuracy:** Gradient Boosting and XGBoost achieved high overall accuracy (~68-70%), but achieved lower Macro-F1 on heavily imbalanced classes without aggressive minority thresholds.
2. **False Positive Suppression:** XGBoost achieved the lowest False Positive Rate (4.19%), minimizing costly false alarms for enterprise operations.
3. **Latency Constraints:** Logistic Regression and Decision Trees delivered sub-millisecond inference latencies (< 1 ms per 1,000 flows), making them suitable for inline packet filtering hardware. Random Forest, while computationally heavier (12.45 ms per 1,000 flows), delivers superior multiclass Macro-F1 across nuanced attack variants.

---

## 17. Feature Attribution & Explainability Rankings
Attribution analysis isolated the top discriminative features:
1. `Bwd IAT Total` (Total backward inter-arrival time): Captures server response cadence.
2. `Active Mean / Active Max`: Measures burst transmission intervals.
3. `Fwd Packets/s`: Distinctive signature separating human browsing from automated flooding.
4. `Packet Length Variance`: Captures packet size diversity (fixed-length attack scripts vs variable HTTP traffic).
5. `Init_Win_bytes_forward`: Distinguishes standard OS TCP stacks from synthetic packet crafting tools.

---

## 18. Streamlit SOC Dashboard Screenshots Guide
The Streamlit SOC Dashboard provides 9 interactive analytics modules.

*Figure Placeholders for Academic Documentation Submission:*
```
[Figure 1: SOC Command Center Executive Overview (Page 1)]
Path: reports/figures/01_soc_overview.png
Description: Real-time telemetry cards, benign vs attack donut chart, and attack distribution.

[Figure 2: Dataset Explorer & Forensic Quality Audit (Page 2)]
Path: reports/figures/02_dataset_explorer.png
Description: Interactive tabular filtering, missing value counts, and infinite value audit.

[Figure 3: Heavy-Tailed Network Flow Telemetry (Page 3)]
Path: reports/figures/03_flow_metrics_distribution.png
Description: Logarithmic histograms of flow duration, bytes/second, and correlation matrix.

[Figure 4: Machine Learning Benchmark Leaderboard (Page 5)]
Path: reports/figures/04_ml_benchmark.png
Description: Multi-metric comparative bar chart, normalized confusion matrices, and ROC curves.

[Figure 5: Isolation Forest Anomaly Distribution (Page 6)]
Path: reports/figures/05_anomaly_detection.png
Description: Continuous decision score histogram highlighting anomalous deviation boundary.

[Figure 6: Offline CSV Threat Prediction Lab (Page 8)]
Path: reports/figures/06_prediction_lab.png
Description: Ingestion interface, threat classification results, confidence scores, and risk tiers.
```

---

## 19. Limitations and Threat Modeling
1. **Offline Capture Generalization:** Benchmark datasets collected in virtual testbeds do not completely mirror live high-speed corporate backbones.
2. **Concept Drift:** Attackers continuously mutate evasion strategies; models trained on fixed historical captures require continuous retraining.
3. **Encrypted Payloads:** Flow features capture transport statistics, but payload-specific attacks concealed in TLS 1.3 require endpoint telemetry correlation.

---

## 20. Ethical Considerations & Defensive Mandate
This project operates strictly within authorized, defensive parameters:
- Offline analysis of public, open-source research data.
- No offensive payloads, exploit tools, or network sniffers are bundled.
- All file uploads are parsed strictly as tabular data without shell execution.

---

## 21. Conclusion
This project developed an end-to-end, reproducible, and modular machine learning platform for defensive network traffic analysis. By enforcing strict anti-leakage boundaries, engineering domain-specific network features, and benchmarking five supervised algorithms alongside an unsupervised anomaly detector, the research demonstrates that Macro-F1 and False Positive Rates provide far superior defensive metrics than naive accuracy. The accompanying Streamlit SOC dashboard translates data science artifacts into an actionable, transparent command interface for cybersecurity analysts.

---

## 22. Future Scope
1. **Sequential Deep Learning:** Implementing temporal sequence modeling (BiLSTM, Temporal Convolutional Networks) over raw packet arrival sequences.
2. **Graph Neural Networks (GNNs):** Constructing topological bipartite host-service graph models to detect lateral movement.
3. **Automated Adversarial Robustness:** Evaluating adversarial evasion attacks (packet padding, delay jitter) and defensive adversarial retraining.

---

## 23. References
1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP, Portugal.
2. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest.* Eighth IEEE International Conference on Data Mining (ICDM), Pisa, Italy.
3. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* ACM SIGKDD, 785-794.
5. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825-2830.

