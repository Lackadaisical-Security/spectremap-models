# Model Cards - Documentation Standard

## Overview

**Every model in this repository MUST have a Model Card.**

Model Cards are standardized documentation that provides transparency about model capabilities, limitations, training data, performance metrics, and ethical considerations.

**Why Model Cards?**
- Transparency: Users understand what models can and cannot do
- Accountability: Developers document their design choices
- Ethics: Biases and limitations are disclosed upfront
- Compliance: Required for responsible AI deployment

**References:**
- [Model Cards for Model Reporting (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993)
- [Google Model Card Toolkit](https://github.com/tensorflow/model-card-toolkit)

---

## Model Card Template

Copy this template for each model. Save as `model_cards/[model_name].md`

````markdown
# Model Card: [Model Name]

**Version:** [e.g., 1.0.0]  
**Date:** [YYYY-MM-DD]  
**Author:** [Your Name / Lackadaisical Security]  
**Contact:** lackadaisicalresearch@pm.me

---

## Model Details

### Model Type
- **Architecture:** [e.g., CNN, LSTM, Transformer, etc.]
- **Framework:** [TensorFlow 2.13+, PyTorch 2.0+, etc.]
- **Task:** [Classification, Regression, Anomaly Detection, etc.]
- **Domain:** [Network Security, Signal Intelligence, Behavioral Analysis, etc.]

### Model Description
[2-3 paragraph description of what the model does and why it was built]

Example:
```
The Anomaly Detector is a convolutional neural network designed to identify
unusual patterns in network traffic. It was built to provide real-time threat
detection for the SpectreMap reconnaissance platform.

The model analyzes time-series network flow data (source/destination IPs, ports,
protocols, packet sizes, timing) and classifies traffic as benign or anomalous.
It is specifically designed to detect port scanning, DDoS attacks, protocol
violations, and unusual communication patterns.
```

### Model Architecture
- **Input Shape:** [e.g., (100, 10) = 100 timesteps, 10 features]
- **Output Shape:** [e.g., (2,) = binary classification]
- **Number of Layers:** [e.g., 5 conv layers + 2 dense layers]
- **Number of Parameters:** [e.g., 1.2M trainable parameters]
- **Model Size:** [e.g., 0.7 MB SavedModel format]

**Architecture Diagram:**
```
Input (100, 10)
  → Conv1D(32 filters, kernel=3)
  → MaxPooling1D(pool_size=2)
  → Conv1D(64 filters, kernel=3)
  → MaxPooling1D(pool_size=2)
  → GlobalAveragePooling1D()
  → Dense(128, activation='relu')
  → Dropout(0.5)
  → Dense(2, activation='softmax')
Output (2,)
```

### Model Metadata
- **License:** [MIT, Apache 2.0, etc.]
- **Repository:** [GitHub URL]
- **Paper:** [arXiv link if applicable]
- **Citation:** [How to cite this model]

---

## Intended Use

### Primary Intended Uses
[List the use cases this model was designed for]

Example:
- ✅ **Authorized penetration testing** - Detecting network reconnaissance during sanctioned engagements
- ✅ **Defensive security monitoring** - Real-time anomaly detection in owned infrastructure
- ✅ **Threat hunting** - Identifying suspicious network behavior in security operations centers
- ✅ **Security research** - Academic/industry research on network intrusion detection

### Out-of-Scope Use Cases
[List uses that are technically possible but NOT recommended]

Example:
- ⚠️ **Unsupervised deployment** - Model requires human review of alerts (not fully autonomous)
- ⚠️ **Non-network domains** - Trained specifically on network traffic (doesn't generalize to other data)
- ⚠️ **Real-time blocking** - Latency and false positive rate require human-in-the-loop

### Prohibited Uses
[List uses that are ILLEGAL or UNETHICAL]

Example:
- ❌ **Unauthorized network monitoring** - Requires legal authorization (pen test contract, network ownership)
- ❌ **Mass surveillance** - Not designed for warrantless surveillance of individuals
- ❌ **Cyber espionage** - Illegal use for state-sponsored hacking or intelligence gathering
- ❌ **Export to sanctioned countries** - Subject to US export controls (see EXPORT_CONTROLS_COMPLIANCE.md)

---

## Training Data

### Dataset Description
[Describe the training data]

Example:
```
**Name:** Synthetic Network Traffic Dataset v1.0
**Source:** Internally generated using network simulation tools
**Size:** 1,000,000 samples (70% benign, 30% anomalous)
**Format:** CSV with 10 features per network flow
**License:** MIT (synthetic, no privacy concerns)
```

### Features
[List and describe input features]

Example:
| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `src_port` | Source port number | Integer | 0-65535 |
| `dst_port` | Destination port number | Integer | 0-65535 |
| `protocol` | IP protocol (TCP/UDP/ICMP) | Categorical | 0-255 |
| `packet_size` | Average packet size (bytes) | Float | 0-1500 |
| `duration` | Flow duration (seconds) | Float | 0-3600 |
| `packet_count` | Number of packets in flow | Integer | 1-100000 |
| ... | ... | ... | ... |

### Data Preprocessing
[Describe preprocessing steps]

Example:
```python
# Normalization
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_raw)

# Categorical encoding
protocol_encoded = OneHotEncoder().fit_transform(protocol)

# Sequence padding
X_padded = pad_sequences(X, maxlen=100, padding='post')
```

### Class Distribution
[Describe label distribution]

Example:
- **Benign Traffic:** 700,000 samples (70%)
- **Port Scan:** 100,000 samples (10%)
- **DDoS Attack:** 80,000 samples (8%)
- **Protocol Violation:** 60,000 samples (6%)
- **Other Anomalies:** 60,000 samples (6%)

### Data Splits
- **Training Set:** 70% (700,000 samples)
- **Validation Set:** 15% (150,000 samples)
- **Test Set:** 15% (150,000 samples)

**Note:** Splits are stratified to maintain class distribution.

### Data Limitations
[Known issues with training data]

Example:
- ⚠️ **Synthetic data** - Generated, not real-world traffic (may not generalize perfectly)
- ⚠️ **Limited attack diversity** - Only 4 attack types (real-world has more)
- ⚠️ **No encrypted traffic** - Doesn't handle TLS/encrypted flows
- ⚠️ **Balanced classes** - Real-world traffic is heavily imbalanced (99%+ benign)

---

## Performance Metrics

### Evaluation Metrics
[Report all relevant metrics - don't cherry-pick]

**Test Set Performance:**
| Metric | Value |
|--------|-------|
| **Accuracy** | 96.3% |
| **Precision (weighted)** | 94.7% |
| **Recall (weighted)** | 95.1% |
| **F1-Score (weighted)** | 94.9% |
| **ROC-AUC** | 0.982 |
| **False Positive Rate** | 2.1% |
| **False Negative Rate** | 4.8% |

**Per-Class Performance:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 98.2% | 97.9% | 98.0% | 105,000 |
| Port Scan | 92.1% | 89.3% | 90.7% | 15,000 |
| DDoS | 91.5% | 93.2% | 92.3% | 12,000 |
| Protocol Violation | 88.7% | 90.1% | 89.4% | 9,000 |
| Other | 87.3% | 85.6% | 86.4% | 9,000 |

### Confusion Matrix
```
                Predicted
              Benign  Attack
Actual Benign  102789   2211  (97.9% recall)
       Attack    3156  41844  (93.0% recall)
```

### Performance Considerations
[Explain performance characteristics]

Example:
- **High recall on attacks** - Designed to catch threats (few false negatives)
- **Moderate false positive rate** - 2.1% FPR acceptable with human review
- **Class imbalance sensitivity** - Performance degrades with extreme imbalance (>99% benign)
- **Inference latency** - 8ms per sample on CPU, 2ms on GPU (suitable for real-time)

---

## Model Limitations

### Known Limitations
[Be honest about what the model CAN'T do]

Example:
1. **Encrypted Traffic** - Cannot analyze encrypted payloads (TLS, VPN)
2. **Novel Attacks** - May not detect zero-day attacks not in training data
3. **Class Imbalance** - Performance degrades on highly imbalanced data (>99% benign)
4. **Feature Engineering** - Requires specific feature extraction from raw packets
5. **Adversarial Vulnerability** - Susceptible to adversarial perturbations (see Security section)

### Failure Modes
[Describe when/how the model fails]

Example:
- **Slow scans** - Stealthy port scans over long time periods may be missed
- **Low-volume attacks** - Small-scale attacks (<10 packets) hard to detect
- **Legitimate anomalies** - Unusual but benign traffic (software updates, backups) triggers false positives
- **Protocol mimicry** - Attacks disguised as legitimate protocols may evade detection

### Bias and Fairness
[Document any biases in the model]

Example:
- **Training data bias** - Synthetic data may not reflect real-world traffic diversity
- **Attack type bias** - Better at detecting port scans (10% of data) than rare attacks (<1%)
- **Network topology bias** - Trained on enterprise networks, may not generalize to IoT/cloud environments
- **Geographic bias** - No geographic variation in training data

### Uncertainty Quantification
[How confident are predictions?]

Example:
```
High confidence (>0.95): 78% of predictions
Medium confidence (0.80-0.95): 18% of predictions
Low confidence (<0.80): 4% of predictions

Recommendation: Human review required for low-confidence predictions
```

---

## Ethical Considerations

### Privacy
[Privacy implications of model use]

Example:
- **Training data privacy** - Synthetic data only, no PII
- **Inference privacy** - Model analyzes network metadata (IPs, ports), not packet payloads
- **Data retention** - Alerts should not store full traffic captures (privacy risk)
- **Anonymization** - IP addresses should be anonymized in logs where possible

### Fairness
[Fairness considerations]

Example:
- **No demographic data** - Model does not use gender, race, age, or other protected attributes
- **Uniform treatment** - All network traffic analyzed equally regardless of source
- **Potential disparate impact** - May flag certain legitimate behaviors (gaming, torrenting) more often

### Accountability
[Who is responsible for model decisions?]

Example:
- **Human-in-the-loop** - Model outputs require human review before action
- **Audit trail** - All predictions logged for accountability
- **Explainability** - Feature importance and attention weights available for interpretation
- **Recourse** - Users can appeal false positives through security operations center

### Dual-Use Concerns
[Can this model be misused?]

Example:
⚠️ **Dual-Use Warning:**
This model has legitimate security applications BUT can also be misused:

**Defensive Use:**
- ✅ Detecting attacks on owned infrastructure
- ✅ Threat hunting with authorization
- ✅ Security research

**Offensive Misuse:**
- ❌ Identifying security tools to evade detection
- ❌ Mapping defenses for attack planning
- ❌ Unauthorized network reconnaissance

**Mitigation:** Model should only be deployed in authorized security contexts.

---

## Security Considerations

### Adversarial Robustness
[How robust is the model to adversarial attacks?]

Example:
**Robustness Testing Results:**
| Attack | Epsilon | Robust Accuracy |
|--------|---------|-----------------|
| FGSM | 0.01 | 87.3% |
| FGSM | 0.05 | 72.1% |
| PGD (10 iter) | 0.05 | 68.4% |
| C&W | - | 81.2% |

**Interpretation:**
- Vulnerable to adversarial perturbations (typical for neural networks)
- FGSM with ε=0.01: 87.3% robust accuracy (12.7% attack success rate)
- Stronger attacks (PGD) reduce accuracy to 68.4%

**Mitigations:**
- Adversarial training included in model training
- Input validation detects extreme perturbations
- Ensemble with other models recommended

### Model Security
[Other security considerations]

Example:
- **Model inversion risk** - LOW (no sensitive training data)
- **Membership inference risk** - LOW (synthetic training data)
- **Backdoor risk** - LOW (trained from scratch, code reviewed)
- **Model stealing risk** - MEDIUM (open source, but adversarial crafting easier)

### Deployment Security
[Secure deployment recommendations]

Example:
1. **Input validation** - Sanitize all inputs before inference
2. **Rate limiting** - Prevent adversarial query flooding
3. **Access control** - Restrict model access to authorized users
4. **Monitoring** - Log all predictions for anomaly detection
5. **Isolation** - Run in sandboxed environment

---

## Export Controls

### Export Classification
[Export control status]

Example:
- **ECCN:** 5D002.c.1 (Information security software - intrusion detection)
- **Alternative ECCN:** 4D004.a (Intrusion software - potential offensive use)
- **Sanctions:** Subject to OFAC comprehensive sanctions (Cuba, Iran, NK, Syria, Russia, Belarus)

### Prohibited Destinations
[Countries where export is prohibited]

See [EXPORT_CONTROLS_COMPLIANCE.md](../EXPORT_CONTROLS_COMPLIANCE.md) for complete list:
- ❌ Cuba, Iran, North Korea, Syria (comprehensively sanctioned)
- ❌ Russia, Belarus (sectoral sanctions - technology prohibited)
- ❌ Arms embargo countries (Venezuela, Myanmar, Libya, etc.)

### Compliance Requirements
[What users must do]

Example:
- Verify recipient not on OFAC SDN List or BIS Denied Persons List
- Ensure end-use is legitimate cybersecurity (not espionage/surveillance)
- Do not export to sanctioned countries
- Maintain records of distribution for 5+ years

---

## Deployment Recommendations

### Hardware Requirements
[Minimum and recommended hardware]

Example:
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | x86-64 | x86-64 with AVX2 |
| **RAM** | 2GB | 8GB+ |
| **GPU** | None (CPU inference) | NVIDIA GPU (CUDA 11.0+) |
| **Storage** | 10MB | SSD preferred |

### Inference Performance
[Latency and throughput]

Example:
- **CPU Inference:** 8ms per sample (125 samples/sec)
- **GPU Inference:** 2ms per sample (500 samples/sec)
- **Batch Inference:** 1200 samples/sec (batch_size=64, GPU)
- **Memory Usage:** 150MB (model + runtime)

### Integration Example
[How to use the model]

```python
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('models/anomaly_detector')

# Prepare input (100 timesteps, 10 features)
X = np.random.randn(1, 100, 10)  # Replace with real data

# Predict
predictions = model.predict(X)
class_id = np.argmax(predictions)
confidence = np.max(predictions)

if class_id == 1 and confidence > 0.8:
    print(f"ALERT: Anomaly detected (confidence: {confidence:.2%})")
else:
    print(f"Benign traffic (confidence: {confidence:.2%})")
```

---

## Maintenance and Updates

### Versioning
[How versions are managed]

Example:
- **Semantic Versioning:** MAJOR.MINOR.PATCH
  - MAJOR: Breaking changes (architecture, input format)
  - MINOR: New features, performance improvements
  - PATCH: Bug fixes, security patches
- **Current Version:** 1.0.0
- **Release Date:** 2026-01-28

### Update Schedule
[When to expect updates]

Example:
- **Security Patches:** As needed (immediate for critical vulnerabilities)
- **Performance Updates:** Quarterly
- **Feature Additions:** Semi-annually
- **Retrain with new data:** Annually

### Monitoring and Feedback
[How to report issues]

Example:
- **Bug Reports:** GitHub Issues
- **Performance Issues:** Email lackadaisicalresearch@pm.me with metrics
- **Security Vulnerabilities:** Email (DO NOT open public issues)
- **Feature Requests:** GitHub Discussions

---

## References

### Papers
[Relevant research papers]

Example:
1. Original architecture: [Paper Title](https://arxiv.org/abs/XXXX.XXXXX)
2. Adversarial training method: [Paper Title](https://arxiv.org/abs/XXXX.XXXXX)
3. Network anomaly detection survey: [Paper Title](https://arxiv.org/abs/XXXX.XXXXX)

### Code
[Related implementations]

Example:
- Training code: `src/spectremap_models/models/anomaly_detector.py`
- Example usage: `examples/export_for_spectremap.py`
- Tests: `tests/test_anomaly_detector.py`

### Datasets
[Public datasets used for comparison]

Example:
- NSL-KDD: [Link](http://www.unb.ca/cic/datasets/nsl.html)
- CICIDS2017: [Link](https://www.unb.ca/cic/datasets/ids-2017.html)
- UNSW-NB15: [Link](https://www.unsw.adfa.edu.au/australian-centre-for-cyber-security/cybersecurity/ADFA-NB15-Datasets/)

---

## Changelog

### Version 1.0.0 (2026-01-28)
- Initial release
- CNN architecture with 1.2M parameters
- 96.3% accuracy on test set
- Adversarial training with FGSM

### Version 0.9.0 (2026-01-15)
- Beta release for internal testing
- 94.1% accuracy
- No adversarial training

---

## Contact

**Model Author:** Lackadaisical Security  
**Email:** lackadaisicalresearch@pm.me  
**Website:** https://lackadaisical-security.com  
**GitHub:** https://github.com/Lackadaisical-Security/spectremap-models

---

## Acknowledgments

[Credit contributors, data sources, inspiration]

Example:
- Training infrastructure provided by [Cloud Provider]
- Dataset generation inspired by [Project Name]
- Architecture based on research by [Author et al.]

---

**Model Card Version:** 1.0  
**Last Updated:** 2026-01-28  
**Copyright © 2025-2026 Lackadaisical Security. All rights reserved.**