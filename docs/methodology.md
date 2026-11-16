# Research Methodology: Machine Learning for Defensive Network Traffic Analysis

## 1. Research Objectives and Problem Formulation
In modern enterprise environments, Network Intrusion Detection Systems (NIDS) must sift through millions of continuous bidirectional flow records per hour. The central research challenge is twofold:
1. **Severe Class Imbalance:** Benign user traffic constitutes ~90-99% of flows, while malicious threats represent rare anomalies.
2. **Adversarial and Non-Stationary Dynamics:** Attackers evolve reconnaissance and evasion techniques (timing jitter, payload encryption, fragmented requests).

This project formalizes the defense problem through both:
- **Supervised Classification:** Mapping learned decision boundaries over known attack families (DDoS, DoS Hulk, PortScan, Brute Force Patator, Botnet C2).
- **Unsupervised Anomaly Detection:** Establishing baseline behavioral boundaries over uninfected normal traffic to detect zero-day deviations.

---

## 2. Experimental Data Hygiene & Anti-Leakage Protocol
A critical vulnerability in published NIDS literature is **data leakage**, wherein models achieve artificial 99.9% accuracy by memorizing dataset-specific artifacts (IP addresses, MAC addresses, timestamps, specific port numbers) rather than learning invariant transport signatures.

### Leakage Prevention Strategy:
1. **Identifier Stripping:** Features including `Flow ID`, `Source IP`, `Destination IP`, `Timestamp`, and capture-specific indices are explicitly dropped prior to modeling.
2. **Strict Partitioning Before Transformations:**
   - The dataset is split into training (75%) and holdout test (25%) partitions using **stratified sampling**.
   - All statistical transformations—including median imputation (`SimpleImputer`), continuous clipping thresholds, constant feature filters, and feature scalers (`StandardScaler`)—are **fitted exclusively on training observations (`X_train`)**.
   - The test set (`X_test`) is transformed using the frozen parameters.
3. **No Test Resampling:** Class balancing techniques (e.g., class weight penalization) are applied during training cost functions only; the test set distribution is never artificially resampled.

---

## 3. Evaluated Machine Learning Paradigms

### A. Baseline Model: Stratified Dummy Classifier
- Serves as the scientific floor. Predicts classes according to observed training prior probabilities. Demonstrates why overall accuracy without context is meaningless.

### B. Linear Baseline: Logistic Regression (L2 Regularized)
- Evaluates the degree of linear separability in standardized feature space.
- Implements class-weighted log-loss optimization to prevent majority class dominance.

### C. Interpretable Tree Model: Decision Tree (CART)
- Employs Gini impurity minimization with constrained depth (`max_depth=15`) and leaf constraints (`min_samples_leaf=5`) to prevent overfitting.
- Provides transparent Boolean decision paths.

### D. Ensemble Model: Random Forest Classifier
- Constructs an ensemble of 100 de-correlated decision trees using bootstrap aggregation (bagging) and random subspace feature sampling.
- Provides robust resistance to flow metric outliers and variance.

### E. Boosting Ensembles: Gradient Boosting & XGBoost
- Sequentially fits shallow decision trees to the negative gradient of the loss function.
- Iteratively prioritizes challenging, borderline traffic flows.

### F. Unsupervised Anomaly Detection: Isolation Forest
- Evaluates outlier properties: anomalies are few and attribute-different.
- Fitted strictly on verified benign baseline communications.
- Assigns anomaly scores based on the average path length required to isolate a flow in randomized partitioning trees.

---

## 4. Evaluation Metrics & Cybersecurity Operational Rationale
Conventional accuracy is unsuited for imbalanced cybersecurity evaluations. A model predicting all flows as benign in a 99:1 dataset achieves 99% accuracy but fails 100% of intrusion attempts.

We adopt the following primary evaluation metrics:

$$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K F1_k$$

$$\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}$$

$$\text{False Negative Rate (FNR)} = \frac{FN}{FN + TP}$$

- **False Positive Rate (FPR):** Measures the rate at which benign traffic is mistakenly flagged as hostile. In enterprise SOC environments, high FPR causes analyst alert fatigue and system downtime.
- **False Negative Rate (FNR):** Measures missed attacks that bypass perimeter inspection.
- **Selection Metric:** Models are ranked and selected primarily by **Macro-F1**, balancing precision and recall equitably across both high-volume floods and low-volume stealth attacks.

---

## 5. Model Explainability vs. Causality
Feature importance is evaluated via:
1. **Gini Impurity (MDI):** Measures total variance reduction contributed by each feature across tree splits.
2. **Permutation Importance:** Measures the degradation in holdout Macro-F1 when a feature's values are randomly shuffled.

> **Academic Disclaimer:** High feature importance indicates strong predictive association within the dataset's statistical distribution. It does NOT establish direct causality. In production environments, attackers can manipulate individual transport features (e.g. packet padding or transmission delays), necessitating multi-layered defense-in-depth.

