# Dataset Documentation Standard

## Overview

**Every training dataset MUST be documented.**

Dataset documentation ensures:
- **Legal compliance** - Verify licensing and permissions
- **Ethical use** - Respect privacy and consent
- **Reproducibility** - Others can verify your results
- **Bias transparency** - Known biases are disclosed
- **Export controls** - Datasets may be export-controlled

**Why This Matters:**
- Training on unauthorized data = legal liability
- PII in training data = privacy violations
- Undocumented bias = discriminatory models
- Poor data quality = poor model performance

---

## Required Documentation Sections

Every dataset documentation file must include these sections. Save as `datasets/[dataset_name]/README.md`

### 1. Dataset Summary

**What to include:**
- **Name and Version** - Dataset name and version number
- **Description** - 2-3 paragraph overview
- **Key Statistics** - Total samples, features, classes, file size
- **Format** - CSV, JSON, TFRecord, etc.
- **Intended Use** - What is this dataset designed for?

**Example:**

> **Dataset:** Synthetic Network Traffic Dataset v1.0.0
> 
> **Description:** Large-scale collection of simulated network flows for training intrusion detection systems. Contains 1,000,000 labeled samples representing benign traffic and various attack types. Generated using network simulation to avoid privacy concerns with real-world captures. All data is synthetic—no real IP addresses, users, or PII.
>
> **Statistics:**
> - Total Samples: 1,000,000
> - Features: 10 numerical features
> - Classes: 2 (benign, anomaly)
> - File Size: 2.3 GB compressed, 8.1 GB uncompressed
> - Format: CSV
>
> **Intended Use:** Training network anomaly detection models for SpectreMap

---

### 2. Dataset Source

**What to include:**
- **Source Type** - Synthetic, public dataset, self-collected, licensed, etc.
- **Generation/Collection Method** - How was the data created or gathered?
- **Legal Basis** - What gives you the right to use this data?
- **License** - MIT, CC-BY, Apache, etc.

**Source Type Options:**
- ✅ **Synthetic** - Artificially generated (no privacy concerns)
- ✅ **Public Dataset** - Publicly available with permissive license
- ✅ **Self-Collected** - Captured with proper authorization
- ⚠️ **Licensed Dataset** - Requires commercial license
- ❌ **Scraped/Unauthorized** - NOT ALLOWED

**Example (Synthetic):**

> **Source Type:** Synthetic
>
> **Generation Method:**
> 1. Define traffic patterns (benign browsing, email, file transfer)
> 2. Generate benign flows using statistical models
> 3. Inject attack signatures (port scans, DDoS, protocol violations)
> 4. Add realistic noise and variation
> 5. Label all samples
> 6. Validate statistical properties
>
> **Tools Used:** Scapy (packet generation), NumPy (statistical modeling), Pandas (data manipulation)
>
> **Legal Basis:** Synthetic data, no legal restrictions
>
> **License:** MIT License

**Example (Self-Collected):**

> **Source Type:** Self-Collected
>
> **Collection Method:**
> - Authorization: Written permission from network owner
> - Scope: Enterprise network (192.168.0.0/16)
> - Duration: 30 days (January 1-30, 2026)
> - Method: Passive packet capture using tcpdump
> - Anonymization: All IP addresses, MACs, and hostnames replaced
> - PII Removal: Packet payloads discarded, only metadata retained
>
> **Legal Basis:** Written authorization, anonymized, privacy-preserving
>
> **License:** MIT License

---

### 3. Privacy and Consent

**What to include:**
- **PII Status** - Does this contain personally identifiable information?
- **Consent** - Was consent obtained (if applicable)?
- **Privacy-Preserving Techniques** - Anonymization, differential privacy, etc.

**Example (Synthetic - No PII):**

> **Contains PII:** NO
>
> This dataset is entirely synthetic. There are no real:
> - IP addresses (all from RFC 1918 private ranges)
> - MAC addresses (randomly generated)
> - Hostnames (fictional: host001, host002, etc.)
> - Usernames, emails, phone numbers (none)
> - Geolocation data (none)
>
> **GDPR Compliance:** Yes (no personal data)
>
> **Consent Required:** No (no human subjects)

**Example (Real Data - Anonymized):**

> **Contains PII:** NO (after anonymization)
>
> **Original Data Contained:**
> - Real IP addresses
> - Real MAC addresses  
> - Hostnames
>
> **Anonymization Process:**
> 1. Replace all IPs with synthetic IPs (consistent mapping preserved)
> 2. Replace all MACs with random MACs
> 3. Replace all hostnames with generic identifiers
> 4. Discard packet payloads (only metadata retained)
> 5. Verify no PII leakage using automated tools
>
> **Privacy Techniques:** Data minimization, k-anonymity, aggregation
>
> **Consent:** Obtained via network usage policy

---

### 4. Dataset Composition

**What to include:**
- **Features** - Table describing each column/feature
- **Class Distribution** - Distribution of labels
- **Data Splits** - Train/val/test splits
- **Statistical Properties** - Key statistics

**Feature Table Example:**

| Feature Name | Type | Range/Values | Description |
|-------------|------|--------------|-------------|
| timestamp | Float | 0.0 - 3600.0 | Time since start of capture (seconds) |
| src_ip | String | 192.168.x.x | Source IP address (synthetic) |
| dst_ip | String | 192.168.x.x | Destination IP address (synthetic) |
| src_port | Integer | 0 - 65535 | Source port number |
| dst_port | Integer | 0 - 65535 | Destination port number |
| protocol | Integer | 6, 17, 1 | IP protocol (TCP/UDP/ICMP) |
| packet_count | Integer | 1 - 100000 | Number of packets in flow |
| byte_count | Integer | 40 - 1500000 | Total bytes transferred |
| duration | Float | 0.0 - 3600.0 | Flow duration (seconds) |
| label | Integer | 0, 1 | 0=benign, 1=attack |

**Class Distribution Example:**

> Total samples: 1,000,000
>
> - Benign: 700,000 (70.0%)
> - Port Scan: 100,000 (10.0%)
> - DDoS Attack: 80,000 (8.0%)
> - Protocol Violation: 60,000 (6.0%)
> - Unusual Traffic: 60,000 (6.0%)
>
> **Note:** Distribution is artificially balanced. Real-world traffic is typically 99%+ benign.

**Data Splits Example:**

> Standard split:
> - Training: 700,000 samples (70%)
> - Validation: 150,000 samples (15%)
> - Test: 150,000 samples (15%)
>
> **Split Methodology:**
> - Stratified (maintains class distribution)
> - Shuffled with random seed (seed=42 for reproducibility)
> - Temporal ordering NOT preserved
>
> **Files:**
> - train.csv
> - val.csv
> - test.csv

---

### 5. Bias and Limitations

**What to include:**
- **Known Biases** - Document all biases in the dataset
- **Limitations** - What this dataset CANNOT do
- **Recommended Mitigations** - How to address limitations

**Example:**

> **Known Biases:**
>
> 1. **Synthetic Bias** - Generated data may not capture real-world edge cases
> 2. **Attack Type Bias** - Port scans over-represented (10% vs <0.1% in reality)
> 3. **Network Topology Bias** - Modeled on enterprise networks only
> 4. **Geographic Bias** - No geographic or time-zone variation
> 5. **Temporal Bias** - No seasonal, day/night, or weekend patterns
>
> **Limitations:**
>
> - ❌ Not real traffic (synthetic generation cannot perfectly replicate reality)
> - ❌ Limited attack diversity (only 4 types, missing SQL injection, XSS, malware, etc.)
> - ❌ No encrypted traffic (TLS/SSL not included)
> - ❌ Balanced classes (70/30 split is artificial, real-world is 99.9/0.1)
> - ❌ No payload data (only metadata, cannot detect payload-based attacks)
>
> **Mitigations:**
>
> 1. Test on real data before deployment
> 2. Fine-tune models on real-world traffic
> 3. Adjust class weights for imbalance
> 4. Augment with real attack samples
> 5. Continuous retraining with new patterns

---

### 6. Quality Assurance

**What to include:**
- **Data Validation** - How was quality verified?
- **Data Cleaning** - What cleaning was performed?
- **Known Issues** - Document any known problems

**Example:**

> **Validation Checks:**
> - ✅ No missing values
> - ✅ No duplicate samples
> - ✅ Valid ranges (IPs, ports, etc.)
> - ✅ Logical consistency (packet_count > 0, byte_count > min_size, etc.)
> - ✅ Label accuracy (manual review: 99.8% correct)
> - ✅ Statistical sanity checks
>
> **Cleaning Steps:**
> 1. Remove samples with missing fields (0 removed)
> 2. Remove duplicates (142 removed)
> 3. Remove outliers (> 5 sigma: 37 removed)
> 4. Fix label errors (18 corrected)
> 5. Normalize feature names
>
> **Known Issues:**
> - Issue #1: Small number of port scans (0.02%) have duration > 10 seconds (generation bug, fixed in v1.1)
> - Issue #2: Some ICMP flows have incorrect byte counts (<0.1% of samples, workaround: filter during preprocessing)

---

### 7. Usage Instructions

**What to include:**
- **Loading Code** - Python examples for loading data
- **Preprocessing** - Normalization, feature engineering
- **Example Training Script** - Complete example

**Loading Examples:**

**Pandas:**
```python
import pandas as pd

train_df = pd.read_csv('datasets/synthetic_traffic/train.csv')
val_df = pd.read_csv('datasets/synthetic_traffic/val.csv')
test_df = pd.read_csv('datasets/synthetic_traffic/test.csv')

print(f"Training samples: {len(train_df)}")
```

**NumPy:**
```python
import numpy as np

data = np.genfromtxt('datasets/synthetic_traffic/train.csv', 
                     delimiter=',', skip_header=1)
X = data[:, :-1]  # Features
y = data[:, -1]   # Labels
```

**TensorFlow:**
```python
import tensorflow as tf

def parse_csv(line):
    defaults = [[0.0]] * 10 + [[0]]
    fields = tf.io.decode_csv(line, record_defaults=defaults)
    features = tf.stack(fields[:-1])
    label = fields[-1]
    return features, label

dataset = tf.data.TextLineDataset('datasets/synthetic_traffic/train.csv')
dataset = dataset.skip(1).map(parse_csv).batch(32)
```

**Preprocessing Example:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler for deployment
import joblib
joblib.dump(scaler, 'models/scaler.pkl')
```

---

### 8. Export Controls

**What to include:**
- **Export Classification** - Is this dataset export-controlled?
- **Rationale** - Why or why not?
- **Compliance Requirements** - What users must do

**Example (Synthetic - Not Controlled):**

> **Export Status:** NOT CONTROLLED
>
> **Rationale:**
> - Synthetic data (no real network information)
> - No PII or sensitive information
> - No classified or confidential data
> - Publicly available under MIT license
>
> **Note:** Models TRAINED on this data may be export-controlled. See EXPORT_CONTROLS_COMPLIANCE.md.

**Example (Real Data - Potentially Controlled):**

> **Export Status:** POTENTIALLY CONTROLLED
>
> **Rationale:**
> - Contains real network traffic metadata (anonymized)
> - Could be "technical data" under EAR
> - May reveal network architectures
>
> **Compliance Requirements:**
> - Do not export to sanctioned countries (Cuba, Iran, NK, Syria, Russia, Belarus)
> - Screen recipients against OFAC SDN and BIS Entity List
> - Maintain distribution records for 5+ years

---

### 9. Versioning and Citation

**What to include:**
- **Version History** - Changes between versions
- **Future Updates** - Planned improvements
- **Citation** - How to cite this dataset

**Version History Example:**

> **Version 1.0.0 (2026-01-28):**
> - Initial release
> - 1,000,000 samples
> - 4 attack types
>
> **Version 0.9.0 (2026-01-15):**
> - Beta release
> - 500,000 samples
> - 2 attack types

**Citation Example:**

**BibTeX:**
```bibtex
@dataset{synthetic_traffic_2026,
  author = {Lackadaisical Security},
  title = {Synthetic Network Traffic Dataset for Intrusion Detection},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/Lackadaisical-Security/spectremap-models},
  license = {MIT}
}
```

**APA:**
Lackadaisical Security. (2026). Synthetic Network Traffic Dataset for Intrusion Detection (Version 1.0.0) [Data set]. GitHub. https://github.com/Lackadaisical-Security/spectremap-models

---

### 10. Contact and Acknowledgments

**What to include:**
- **Curator Contact** - Email, GitHub, etc.
- **Support Channels** - Where to ask questions
- **Acknowledgments** - Credit sources, tools, contributors

**Example:**

> **Dataset Curator:** Lackadaisical Security  
> **Email:** lackadaisicalresearch@pm.me  
> **GitHub Issues:** https://github.com/Lackadaisical-Security/spectremap-models/issues
>
> **For questions about:**
> - Dataset bugs → GitHub Issues
> - Licensing → Email
> - Privacy concerns → Email
>
> **Acknowledgments:**
> - Dataset generation inspired by CICIDS2017 methodology
> - Simulation tools: Scapy, NumPy, Pandas
> - Thanks to [Contributors] for feedback

---

## Checklist for Dataset Documentation

Before releasing a dataset, verify:

- [ ] All 10 sections filled out completely
- [ ] Legal status verified (license, permissions, consent)
- [ ] Privacy implications documented (PII status, anonymization)
- [ ] Bias and limitations disclosed honestly
- [ ] Quality assurance performed (validation, cleaning)
- [ ] Usage instructions provided (loading, preprocessing examples)
- [ ] Export controls considered
- [ ] Citation information included
- [ ] Contact information current
- [ ] Known issues documented

---

## Prohibited Datasets

**DO NOT contribute datasets that:**
- ❌ Contain unauthorized PII
- ❌ Were obtained illegally (hacking, scraping without permission)
- ❌ Violate copyright or license terms
- ❌ Contain classified or confidential information
- ❌ Were collected from sanctioned countries (export control risk)
- ❌ Violate GDPR, CCPA, or other privacy regulations
- ❌ Lack proper consent (for human subject data)

**If in doubt, ask:** lackadaisicalresearch@pm.me

---

## Example Dataset Documentation

See `datasets/synthetic_traffic/README.md` for a complete example following this standard.

---

## Additional Resources

**Privacy & Ethics:**
- GDPR Compliance: https://gdpr.eu/
- CCPA Compliance: https://oag.ca.gov/privacy/ccpa
- AI Ethics Guidelines: https://www.partnershiponai.org/

**Data Quality:**
- Data Validation Best Practices
- Statistical Testing for Data Quality
- Bias Detection Tools

**Export Controls:**
- See EXPORT_CONTROLS_COMPLIANCE.md
- BIS Export Regulations: https://www.bis.doc.gov/

---

### 🔥 **Documented by Lackadaisical Security** 🔥

*"Data is the lifeblood of artificial intelligence. Corrupt the data, corrupt the model. Bias the data, bias the world. Steal the data, face the law. Document your sources with precision. Respect privacy with diligence. Disclose limitations with honesty. Only then can we forge AI that serves humanity rather than exploits it."*

— **Lackadaisical Security, The Operator** (2025)

**Copyright © 2025-2026 Lackadaisical Security. All rights reserved.**