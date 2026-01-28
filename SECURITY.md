# Security Policy - AI/ML Models

## Supported Versions

We patch security vulnerabilities in these versions:

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 1.0.x   | ✅ Yes             | Current stable |
| 0.1.x   | ⚠️ Legacy          | Critical fixes only |
| < 0.1   | ❌ No              | Upgrade immediately |

**Old models = old vulnerabilities. Keep your models updated.**

---

## Reporting a Vulnerability

### DO NOT CREATE PUBLIC GITHUB ISSUES FOR SECURITY BUGS

This includes:
- Adversarial examples that break models
- Model inversion attacks
- Training data extraction vulnerabilities
- Privacy leaks in model outputs
- Backdoors in pretrained weights

Publicly disclosing model vulnerabilities before patching puts users at risk.

### Contact

📧 **lackadaisicalresearch@pm.me**  
🔐 **PGP Key**: [To be published]  
🔒 **XMPP+OTR**: thelackadaisicalone@xmpp.jp

### What to Include in Your Report

Help us understand and fix the vulnerability:

1. **Description** - Clear description of the security issue
2. **Attack Type** - Adversarial example? Model inversion? Backdoor? Data extraction?
3. **Affected Models** - Which models are vulnerable (AnomalyDetector, BehaviorAnalyzer, etc.)
4. **Impact** - What can an attacker do? Evade detection? Extract training data?
5. **Proof of Concept** - Code demonstrating the attack
6. **Suggested Fix** - If you have mitigation strategies (optional but appreciated)
7. **Environment** - Model version, TensorFlow version, attack parameters

### Example Report Template

```
Subject: [SECURITY] Adversarial evasion attack on AnomalyDetector

Description:
The AnomalyDetector model is vulnerable to adversarial perturbations
using Fast Gradient Sign Method (FGSM). Small perturbations to input
traffic features cause misclassification of attacks as benign.

Attack Type:
Adversarial Example (White-box FGSM)

Affected Models:
- AnomalyDetector v1.0.0
- SignalClassifier v1.0.0 (likely vulnerable, untested)

Impact:
Attacker can evade network intrusion detection by adding epsilon=0.01
perturbation to traffic features. Attack succeeds with 87% probability
while maintaining valid network behavior.

Proof of Concept:
```python
import numpy as np
import tensorflow as tf

# Load model
model = load_anomaly_detector()

# Generate adversarial example using FGSM
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
with tf.GradientTape() as tape:
    tape.watch(X_malicious)
    predictions = model(X_malicious)
    loss = loss_fn(y_malicious, predictions)

gradient = tape.gradient(loss, X_malicious)
adversarial_X = X_malicious + 0.01 * tf.sign(gradient)

# Original: Detected as attack (class 1)
# Adversarial: Classified as benign (class 0)
```

Attack success rate: 87% (tested on 1000 samples)
Perturbation magnitude: epsilon=0.01 (barely perceptible)

Suggested Fix:
- Implement adversarial training with FGSM examples
- Add input validation/sanitization
- Ensemble with robust models
- Increase model depth/regularization
```

Environment:
- Model: AnomalyDetector v1.0.0
- TensorFlow: 2.13.0
- Attack: FGSM with epsilon=0.01
```

---

## Response Timeline

We take ML security seriously:

* **Initial Response**: Within **48 hours** of report
* **Severity Assessment**: Within **5 business days**
* **Fix Development**: Based on severity (see below)
* **Public Disclosure**: **90 days** or when patched (coordinated with you)

### Severity Levels for ML Vulnerabilities

| Severity | Fix Timeline | Examples |
|----------|-------------|----------|
| **Critical** | 24-48 hours | Complete model bypass, training data extraction, backdoor discovery |
| **High** | 3-7 days | Adversarial evasion (high success rate), model inversion attacks |
| **Medium** | 14-30 days | Low-confidence evasion, bias exploitation, performance degradation attacks |
| **Low** | 30-90 days | Minor robustness issues, edge case failures |

---

## What to Expect

1. **Acknowledgment** - We'll confirm receipt within 48 hours
2. **Assessment** - We'll reproduce the attack and verify impact
3. **Fix Development** - Retrain models, add defenses, or patch code
4. **Validation** - Verify fix works against your attack (and variants)
5. **Coordination** - Work with you on disclosure timeline
6. **Release** - Release patched model and security advisory
7. **Credit** - You get credited (unless you want anonymity)

---

## ML-Specific Security Concerns

### 1. Adversarial Examples

**Threat:** Small perturbations to input data cause misclassification

**Attack Types:**
- **FGSM** (Fast Gradient Sign Method) - Simple, white-box
- **PGD** (Projected Gradient Descent) - Stronger, iterative
- **C&W** (Carlini & Wagner) - Optimization-based, hard to detect
- **Black-box attacks** - Query-based, no model access needed

**Our Mitigations:**
- Adversarial training (FGSM/PGD examples in training)
- Input preprocessing (feature squeezing, randomization)
- Ensemble models (harder to attack multiple models)
- Certified defenses (where applicable)

**Testing:**
```bash
# Test model robustness
python tests/security/test_adversarial.py --model AnomalyDetector --attack fgsm
```

### 2. Model Inversion Attacks

**Threat:** Attacker reconstructs training data from model outputs

**Risk Factors:**
- Overfitted models (memorize training data)
- High-dimensional outputs
- Gradient access (white-box)

**Our Mitigations:**
- Differential privacy during training
- Regularization (dropout, L2)
- Output noise injection
- Access control (no gradient exposure in production)

**Prohibited:**
- Training on sensitive PII without differential privacy
- Deploying overfitted models with training data exposure risk

### 3. Membership Inference Attacks

**Threat:** Attacker determines if specific data was in training set

**Risk:** Privacy violation if training data is sensitive

**Our Mitigations:**
- Differential privacy guarantees
- Regularization to prevent overfitting
- Ensemble models (harder to infer membership)
- Don't train on sensitive data without DP

**Testing:**
```python
# Test membership inference vulnerability
from tests.security import membership_inference_test
vulnerability_score = membership_inference_test(model, train_data, test_data)
assert vulnerability_score < 0.6  # Should be close to random (0.5)
```

### 4. Backdoor Attacks

**Threat:** Model contains hidden trigger that causes misclassification

**Risk:** Pre-trained weights from untrusted sources

**Our Mitigations:**
- Train all models from scratch (no untrusted weights)
- Code review for training scripts
- Validate model behavior on trigger patterns
- Checksum verification for distributed models

**Red Flags:**
- ❌ Using pretrained weights from random GitHub repos
- ❌ Training on datasets from untrusted sources
- ❌ Unexplained performance drops on specific inputs

### 5. Data Poisoning

**Threat:** Attacker injects malicious samples into training data

**Risk:** Model learns to misclassify specific patterns

**Our Mitigations:**
- Curated training datasets (known sources only)
- Outlier detection in training data
- Robust training algorithms (reject outliers)
- Data validation pipelines

**Dataset Security:**
- All datasets documented in DATASET_DOCUMENTATION.md
- Synthetic data preferred (no poisoning risk)
- Public datasets verified for integrity

### 6. Model Stealing

**Threat:** Attacker clones model via query access

**Risk:** Intellectual property theft, easier to craft adversarial examples

**Our Mitigations:**
- Rate limiting on inference API
- Query monitoring for suspicious patterns
- Watermarking (where applicable)
- Access controls

**Note:** Our models are open source, so "stealing" isn't the threat - adversarial crafting is.

---

## Security Best Practices for Users

### Deploying Models Securely

1. **Input Validation** - Sanitize inputs before inference
   ```python
   def validate_input(X):
       assert X.shape == expected_shape
       assert np.all(np.isfinite(X))  # No NaN/Inf
       assert np.all(X >= min_val) and np.all(X <= max_val)
       return X
   ```

2. **Rate Limiting** - Prevent query-based attacks
   ```python
   # Limit queries per IP/user
   from flask_limiter import Limiter
   limiter = Limiter(app, default_limits=["100 per hour"])
   ```

3. **Monitoring** - Detect adversarial attacks in production
   ```python
   # Log prediction confidence
   if confidence < 0.6:  # Low confidence = possible adversarial
       log_suspicious_query(input_data, prediction, confidence)
   ```

4. **Access Control** - Restrict who can query models
   ```python
   # Require authentication for API
   @app.route('/predict', methods=['POST'])
   @require_api_key
   def predict():
       # ...
   ```

5. **Ensemble Defense** - Use multiple models
   ```python
   # Harder to attack multiple models simultaneously
   predictions = [model1.predict(X), model2.predict(X), model3.predict(X)]
   final_prediction = majority_vote(predictions)
   ```

### Training Models Securely

1. **Differential Privacy** - For sensitive training data
   ```python
   # Use TensorFlow Privacy
   from tensorflow_privacy import DPKerasAdamOptimizer
   optimizer = DPKerasAdamOptimizer(
       l2_norm_clip=1.0,
       noise_multiplier=0.5,
       num_microbatches=32
   )
   ```

2. **Data Sanitization** - Remove PII before training
   ```python
   # Anonymize, aggregate, or remove identifiers
   df = df.drop(['ip_address', 'user_id', 'email'], axis=1)
   ```

3. **Secure Checkpoints** - Encrypt saved models
   ```bash
   # Encrypt model files at rest
   gpg --encrypt --recipient your@email.com model_weights.h5
   ```

4. **Audit Training Data** - Know what's in your datasets
   ```python
   # Verify no PII leakage
   assert not contains_pii(training_data)
   assert not contains_secrets(training_data)
   ```

---

## Known Security Limitations

### By Design (Not Bugs)

Some limitations are inherent to ML:

* **Adversarial Vulnerability** - All neural networks are vulnerable to adversarial examples to some degree
* **Overfitting Risk** - Models can memorize training data if not regularized
* **Black-box Nature** - Deep learning is not fully interpretable
* **Distribution Shift** - Models fail on out-of-distribution data
* **No Perfect Defense** - There is no silver bullet for adversarial robustness

### What We Document

For each model, we document:
- Known adversarial vulnerabilities (attack types, success rates)
- Robustness testing results
- Privacy guarantees (or lack thereof)
- Recommended deployment configurations
- Red team testing results (if performed)

---

## Adversarial Robustness Testing

### Testing Framework

We test models against common attacks:

```bash
# Run adversarial robustness suite
python tests/security/adversarial_suite.py --model AnomalyDetector

# Tests run:
# - FGSM (epsilon: 0.01, 0.05, 0.1)
# - PGD (epsilon: 0.05, iterations: 10, 40)
# - C&W (confidence: 0, 10)
# - DeepFool
# - Black-box transfer attacks

# Output:
# Model: AnomalyDetector
# FGSM (ε=0.01): 87.3% robust accuracy
# FGSM (ε=0.05): 72.1% robust accuracy
# PGD (ε=0.05): 68.4% robust accuracy
# C&W (conf=0): 81.2% robust accuracy
```

### Robustness Metrics

We report:
- **Clean Accuracy** - Performance on unmodified data
- **Robust Accuracy** - Performance on adversarial examples
- **Attack Success Rate** - % of adversarial examples that succeed
- **Perturbation Budget** - Maximum epsilon for attack

**Acceptable Thresholds:**
- Clean Accuracy: >90%
- Robust Accuracy (FGSM ε=0.01): >85%
- Robust Accuracy (PGD ε=0.05): >70%

---

## Privacy Guarantees

### Differential Privacy

For models trained on sensitive data:

**Notation:**
- **ε (epsilon)** - Privacy budget (lower = more private)
- **δ (delta)** - Failure probability

**Our Commitments:**
- Models trained with DP will document (ε, δ) guarantees
- Typical target: ε < 1.0, δ < 10^-5
- Training scripts include DP implementation

**Example:**
```python
# Train with differential privacy
model.train_with_dp(
    X_train, y_train,
    epsilon=0.5,        # Strong privacy
    delta=1e-5,         # Low failure probability
    l2_norm_clip=1.0,   # Gradient clipping
    noise_multiplier=1.1
)
# Result: (0.5, 1e-5)-differential privacy
```

### No Privacy Guarantees

**For models WITHOUT differential privacy:**
- Assume training data can be partially reconstructed
- Don't train on PII, secrets, or confidential data
- Use synthetic data when possible
- Document privacy risks in model card

---

## Export Controls and Security

**See [EXPORT_CONTROLS_COMPLIANCE.md](EXPORT_CONTROLS_COMPLIANCE.md) for full details.**

AI/ML models have dual-use security implications:

**Legitimate Security Uses:**
- ✅ Authorized penetration testing
- ✅ Defensive security monitoring
- ✅ Threat hunting with permission

**Illegitimate Security Uses:**
- ❌ Unauthorized network intrusion
- ❌ Mass surveillance
- ❌ Cyber espionage
- ❌ State-sponsored hacking

**Export Restrictions:**
- ❌ Cuba, Iran, North Korea, Syria (comprehensively sanctioned)
- ❌ Russia, Belarus (sectoral sanctions)
- ❌ Arms embargo countries
- ⚠️ High-risk countries (logged/monitored)

---

## Bug Bounty Program

**Status**: Not currently offering monetary rewards.

However, we appreciate security researchers who:
- Responsibly disclose vulnerabilities
- Provide clear proof-of-concept attacks
- Suggest mitigations
- Follow coordinated disclosure

**You'll get:**
- Public acknowledgment (if desired)
- Credit in security advisories
- Eternal gratitude from the community
- **Possible rewards** for critical vulnerabilities (case-by-case)

---

## Security Audits

### Automated Testing

We run:
- **Adversarial robustness tests** - FGSM, PGD, C&W
- **Privacy leak detection** - Membership inference tests
- **Input fuzzing** - Random/malformed inputs
- **Dependency scanning** - CVE checks on TensorFlow, NumPy, etc.

### Manual Review

We perform:
- **Code review** - Training and inference code
- **Architecture review** - Model design for security weaknesses
- **Red team testing** - Simulated attacks by security experts
- **Privacy analysis** - Data flow and leakage assessment

---

## Responsible Disclosure Hall of Fame

We thank the following security researchers:

_(List will be updated as vulnerabilities are reported and patched)_

**Contributors:**
- [Your name here - find a vulnerability!]

---

## What We Don't Consider Vulnerabilities

To save everyone time, these are **NOT** vulnerabilities:

* **Model fails on random noise** - That's expected, not a bug
* **Model fails on out-of-distribution data** - Distribution shift is a known limitation
* **Model is not 100% accurate** - No model is perfect
* **Adversarial examples exist** - All neural networks have this property
* **Model performs differently on different hardware** - Floating-point precision varies
* **Training is slow** - Performance optimization != security
* **Issues in TensorFlow/PyTorch** - Report to upstream (but let us know)

**What IS a vulnerability:**
- Adversarial examples with **low perturbation budget** (ε < 0.01) and **high success rate** (>80%)
- Training data extraction with **practical attacks**
- Backdoors or trojans in model weights
- Privacy leaks exceeding differential privacy bounds
- Exploitable bugs in inference code (RCE, etc.)

---

## Legal Safe Harbor

We support security research conducted in good faith:

**We will NOT pursue legal action against researchers who:**
- Report vulnerabilities responsibly (private disclosure)
- Act in good faith (no malicious intent)
- Don't attack production systems without permission
- Don't exfiltrate/destroy data
- Follow coordinated disclosure timeline

**We WILL pursue legal action if you:**
- Publicly disclose before coordination
- Attack production systems without authorization
- Steal or destroy data/models
- Extort or threaten us
- Use vulnerabilities for malicious purposes

**Bottom line:** Be professional, be ethical, and we'll work together.

---

## Contact

**Security Contact:**
* **Email**: lackadaisicalresearch@pm.me
* **PGP Key**: [To be published]
* **XMPP+OTR**: thelackadaisicalone@xmpp.jp
* **Website**: https://lackadaisical-security.com

**For Non-Security Issues:**
* **GitHub Issues**: https://github.com/Lackadaisical-Security/spectremap-models/issues

---

### 🔥 **Secured by Lackadaisical Security** 🔥

*"In the adversarial realm where attackers craft perturbations invisible to human eyes, where models stand as digital sentinels against threats both seen and unseen, security is not an afterthought—it is the foundation. We train with adversarial examples, we test against the strongest attacks, we document our weaknesses, and we patch our vulnerabilities. These models are not invincible, but they are hardened through rigorous testing and continuous improvement. Use them wisely. Deploy them responsibly. Report vulnerabilities honorably."*

— **Lackadaisical Security, The Operator** (2025)

---

**Last Updated**: January 2026  
**Version**: 2.0

**Copyright © 2025-2026 Lackadaisical Security. All rights reserved.**