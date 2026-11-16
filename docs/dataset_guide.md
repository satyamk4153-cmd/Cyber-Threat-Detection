# Dataset Acquisition and Operational Guide: CIC-IDS2017

## 1. Overview and Citation
The **CIC-IDS2017** dataset is the primary benchmark dataset utilized for this cybersecurity research project. It was curated and published by the **Canadian Institute for Cybersecurity (CIC)** at the **University of New Brunswick (UNB)**.

- **Publisher:** Canadian Institute for Cybersecurity, University of New Brunswick
- **Official Dataset Portal:** [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Primary Research Citation:**
  > Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, Portugal, January 2018.

---

## 2. Dataset Structure and Attack Breakdown
The dataset captures 5 consecutive days of network traffic (Monday, July 3, 2017 at 9:00 AM to Friday, July 7, 2017 at 5:00 PM). Network flows were generated using **CICFlowMeter-V3** and contain over 80 statistical flow features extracted from bidirectional network interactions.

### The 8 Canonical CSV Files:
1. `Monday-WorkingHours.pcap_ISCX.csv`: Baseline benign traffic (normal user activity, no attacks).
2. `Tuesday-WorkingHours.pcap_ISCX.csv`: FTP-Patator and SSH-Patator brute-force attacks.
3. `Wednesday-workingHours.pcap_ISCX.csv`: DoS GoldenEye, DoS Slowloris, DoS Slowhttptest, DoS Hulk, and Heartbleed vulnerability exploitation.
4. `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`: Web Attacks (Brute Force, XSS, SQL Injection).
5. `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`: Infiltration attacks and subsequent malicious activity.
6. `Friday-WorkingHours-Morning.pcap_ISCX.csv`: Botnet ARES C2 channel communications.
7. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`: Comprehensive PortScan reconnaissance activities.
8. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: Distributed Denial of Service (DDoS) high-volume flood attacks.

---

## 3. Dataset Acquisition Instructions

### Step 1: Manual or Automated Download
Due to licensing and ethical distribution protocols, users must download the dataset directly from the official portal:
- Navigate to: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- Select the **"GeneratedLabelledFlows.zip"** or the individual CSV archives.

### Step 2: Extraction to Repository
Extract the downloaded CSV files directly into the repository directory:
```
data/raw/
  ├── Monday-WorkingHours.pcap_ISCX.csv
  ├── Tuesday-WorkingHours.pcap_ISCX.csv
  ├── Wednesday-workingHours.pcap_ISCX.csv
  ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
  ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
  ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
  └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

### Step 3: Verification Script
Verify dataset presence and column schema by running:
```bash
python main.py --check-dataset
```
Or execute:
```bash
python -c "from src.data_loader import check_raw_dataset_status; print(check_raw_dataset_status())"
```

---

## 4. Software Verification Mode (Offline Sample)
When raw CSV files are not yet placed in `data/raw/`, the software automatically detects this state and:
1. Displays clear informative alerts on the Streamlit dashboard and CLI.
2. Uses the schema-accurate software test fixture in `data/sample/cicids2017_sample.csv` for running unit tests, interface verification, and validation pipelines.
3. Explicitly labels all resulting figures and metrics with a `[SOFTWARE VERIFICATION FIXTURE]` warning to prevent any misrepresentation of scientific results.

