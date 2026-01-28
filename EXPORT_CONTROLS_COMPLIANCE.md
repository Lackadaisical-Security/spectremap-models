# Export Controls Compliance Notice - AI/ML Models
## Spectre Map Models Repository

**Document Version:** 1.0  
**Effective Date:** January 28, 2026  
**Repository:** spectremap-models  
**Prepared by:** Lackadaisical Security  

---

## ⚠️ CRITICAL NOTICE

**These AI/ML models are subject to US export control regulations.**

This repository contains trained neural networks and machine learning models specifically designed for cybersecurity operations, including:
- Network intrusion detection
- Behavioral analysis for threat hunting
- Signal intelligence classification
- Anomaly detection in communication patterns

**These capabilities make them DUAL-USE TECHNOLOGIES subject to export controls.**

---

## 1. Executive Summary

The Spectre Map Models repository contains production-ready AI/ML models that are controlled under:

- **Export Administration Regulations (EAR)** - 15 CFR Parts 730-774
- **OFAC Sanctions Programs** - 31 CFR Chapter V
- **Emerging AI/ML Export Controls** - New regulations under development

**Users, developers, and distributors are solely responsible for compliance with all applicable export control laws.**

---

## 2. Why AI/ML Models Are Export Controlled

### 2.1 Traditional Software Controls

All trained models are considered **"software"** under EAR:
- **Compiled models** (SavedModel, .pb, .h5 files) = Software under ECCN 5D002
- **Source code** (training scripts) = Technology under ECCN 5D002 or 5E002
- **Training datasets** = Potentially controlled technical data

### 2.2 AI-Specific Concerns

AI/ML models have unique export control implications:

#### **Intrusion/Surveillance Capabilities (ECCN 4D004)**
- **Anomaly Detector** - Can identify network intrusions, potentially used for surveillance
- **Behavior Analyzer** - Profiles user/entity behavior, dual-use for monitoring
- **Signal Classifier** - Classifies RF signals, applicable to military SIGINT

#### **Cybersecurity Items (ECCN 5D002)**
- Models designed for information security operations
- Threat detection and analysis capabilities
- Integration with offensive security tools

#### **Emerging AI Controls**
The US government is actively developing new controls on:
- **AI for surveillance** - Facial recognition, behavior tracking
- **AI for cyber operations** - Offensive/defensive cyber capabilities
- **Adversarial AI** - Models designed to attack other AI systems
- **Dual-use AI** - Commercial AI with military applications

---

## 3. Applicable US Regulations

### 3.1 Export Administration Regulations (EAR)

**Controlling Agency:** Bureau of Industry and Security (BIS), U.S. Department of Commerce

#### **ECCN 5D002 - Information Security Software**

Spectre Map Models likely fall under **ECCN 5D002.c.1**:
- Software designed for "information security"
- Includes cybersecurity applications
- Includes intrusion detection systems

**Components:**
- **Anomaly Detector** - Network intrusion detection → 5D002.c.1
- **Behavior Analyzer** - Security event correlation → 5D002.c.1
- **Signal Classifier** - Telecommunications analysis → 5D002.c.1 or 5A002

**License Exception:** May qualify for **License Exception ENC** under 15 CFR § 740.17(b) if:
- Models are publicly available (open source), OR
- One-time self-classification report filed with BIS, OR
- Annual sales reports submitted (if commercialized)

#### **ECCN 4D004 - Intrusion Software**

Models may fall under **ECCN 4D004.a** if they:
- Enable unauthorized access to systems
- Defeat protective countermeasures
- Perform covert surveillance
- Extract data without authorization

**Applicability to Spectre Map Models:**
- **Anomaly Detector** - Identifies network defenses (potential 4D004 concern)
- **Behavior Analyzer** - Profiling for reconnaissance (potential 4D004 concern)
- **Signal Classifier** - SIGINT capabilities (potential 4D004 concern)

**Risk Level:** HIGH - These models are designed for offensive security operations.

#### **ECCN 3D001/5A002 - Telecommunications**

**Signal Classifier** model specifically:
- Analyzes RF signals (WiFi, Bluetooth, Zigbee, LTE)
- May fall under telecommunications equipment controls
- Potential military SIGINT applications

### 3.2 OFAC Sanctions (Same as Main Repo)

**Controlling Agency:** U.S. Department of the Treasury, Office of Foreign Assets Control

**Comprehensively Sanctioned Countries (COMPLETE EXPORT PROHIBITION):**
- ❌ **Cuba** (CU) - 31 CFR Part 515
- ❌ **Iran** (IR) - 31 CFR Part 560
- ❌ **North Korea** (KP) - 31 CFR Part 510
- ❌ **Syria** (SY) - 31 CFR Part 542
- ❌ **Crimea/Donetsk/Luhansk** (Ukraine regions) - 31 CFR Part 589

**Sectoral Sanctions (BLOCKED for dual-use technology):**
- ❌ **Russia** (RU) - 31 CFR Part 589
- ❌ **Belarus** (BY) - 31 CFR Part 548

**Arms Embargo Countries (BLOCKED):**
- ❌ Venezuela, Myanmar, Libya, Somalia, South Sudan, Sudan, CAR, Yemen, etc.

**High-Risk Countries (Allowed with logging/monitoring):**
- ⚠️ China, Hong Kong, Pakistan, Afghanistan, Iraq, UAE, Turkey, Egypt

### 3.3 International Traffic in Arms Regulations (ITAR)

**Controlling Agency:** Directorate of Defense Trade Controls (DDTC), U.S. Department of State

**Applicability Assessment:**

Spectre Map Models are **likely NOT ITAR-controlled** unless:
1. Developed under DoD/Intelligence Community contracts
2. Specifically designed for military SIGINT/cyber warfare
3. Classified or controlled under USML Category XI or XIII
4. Explicitly designated as defense articles by US government

**However**, models could become ITAR-controlled if:
- Integrated into military cyber operations platforms
- Used for classified government programs
- Customized for defense/intelligence applications

---

## 4. Model-Specific Export Classifications

### 4.1 Anomaly Detector

**Description:** CNN-based network traffic anomaly detection model

**Primary Use Cases:**
- Port scan detection
- DDoS attack identification
- Protocol violation detection
- Unusual traffic pattern recognition

**Export Control Classification:**
- **Primary ECCN:** 5D002.c.1 (Information security software - intrusion detection)
- **Secondary ECCN:** 4D004.a (If used to identify defenses for attack planning)

**Export Risk:** HIGH
- Dual-use for both defensive and offensive operations
- Can map network defenses (offensive intelligence gathering)
- Military applications in cyber warfare

**Prohibited Uses:**
- Unauthorized network reconnaissance
- Bypassing security controls
- Cyber espionage
- State-sponsored hacking

### 4.2 Behavior Analyzer

**Description:** Bidirectional LSTM model for entity behavioral profiling

**Primary Use Cases:**
- Insider threat detection
- Lateral movement identification
- User behavior anomaly detection
- Device behavior profiling

**Export Control Classification:**
- **Primary ECCN:** 5D002.c.1 (Information security software)
- **Secondary ECCN:** 4D004.a (Surveillance/monitoring capabilities)

**Export Risk:** HIGH
- Can be used for unauthorized surveillance
- Profiling individuals without consent
- State surveillance applications
- Privacy violations

**Prohibited Uses:**
- Mass surveillance
- Political dissident tracking
- Unauthorized monitoring
- Human rights violations

### 4.3 Signal Classifier

**Description:** Deep CNN for RF signal classification (WiFi/BLE/Zigbee/LTE)

**Primary Use Cases:**
- Wireless device identification
- RF spectrum analysis
- Signal intelligence (SIGINT)
- Wireless protocol classification

**Export Control Classification:**
- **Primary ECCN:** 5A002 or 5D002 (Telecommunications equipment/software)
- **Secondary ECCN:** 3D001 (Electronic equipment for signal analysis)
- **Potential ITAR:** USML Category XI (If used for military SIGINT)

**Export Risk:** VERY HIGH
- Direct military SIGINT applications
- Intelligence collection capabilities
- Can identify military/government communications
- Dual-use in electronic warfare

**Prohibited Uses:**
- Military signal intelligence (without authorization)
- Espionage
- Unauthorized spectrum monitoring
- Intercepting government communications

---

## 5. Training Data and Dataset Controls

### 5.1 Training Data as Technical Data

**Training datasets may be controlled as "technical data" under EAR Part 772:**

- **Network traffic captures** - May contain sensitive network architectures
- **Behavioral datasets** - May include personally identifiable information (PII)
- **RF signal samples** - May include military/government signal signatures

**Export Control Implications:**
- Sharing training data = Export of technical data
- May require export license even if models are exempt
- Privacy regulations (GDPR, CCPA) also apply

### 5.2 Synthetic vs. Real-World Data

**Synthetic Data:**
- Artificially generated training data
- Lower export control risk
- No privacy concerns
- Recommended for open-source projects

**Real-World Data:**
- Captured from actual networks/systems
- Higher export control risk
- Privacy and consent issues
- May contain classified/sensitive information

**Spectre Map Models Use:**
- **Anomaly Detector** - Synthetic network traffic (low risk)
- **Behavior Analyzer** - Synthetic user behavior (low risk)
- **Signal Classifier** - Public RF signal samples (medium risk)

---

## 6. Open Source and Public Availability

### 6.1 Public Domain Exclusion

**EAR § 734.7 - Publicly Available Technology and Software**

Models may be excluded from EAR if:
1. **Published** - Made available to the public without restrictions
2. **Unrestricted** - No limitations on further dissemination
3. **Accessible** - Available to anyone without approval

**GitHub Public Repository:**
- Qualifies as "publicly available" if no access restrictions
- Still subject to sanctioned country prohibitions (OFAC)
- Still prohibited to denied parties (SDN List, Entity List)

### 6.2 Deemed Export Considerations

**Sharing models with foreign nationals in the US = Export**

- Allowing non-US developers to access model code/weights
- Collaborating with international researchers
- Hiring foreign nationals to work on models

**Mitigation:**
- Screen collaborators' nationalities
- Implement access controls for model weights
- Document foreign national access

---

## 7. Compliance Procedures

### 7.1 Pre-Distribution Checklist

Before distributing trained models or code:

- [ ] **Classify Models** - Determine ECCN for each model
- [ ] **Screen Recipients** - Check against OFAC SDN, BIS Denied Persons, BIS Entity List
- [ ] **Verify Destination** - Ensure country is not sanctioned
- [ ] **Assess End-Use** - Verify legitimate cybersecurity purpose
- [ ] **License Check** - Determine if export license required
- [ ] **Document Transaction** - Record who, what, when, where, why
- [ ] **Encryption Notice** (if applicable) - File BIS notification for encryption
- [ ] **Implement Technical Controls** - Geoblocking for sanctioned countries

### 7.2 GitHub-Specific Compliance

**GitHub Export Control Features:**
- Automatically blocks access from Cuba, Iran, North Korea, Syria, Crimea
- Complies with US trade controls
- Developer remains responsible for compliance

**Additional Steps:**
1. **Include EXPORT_CONTROLS_COMPLIANCE.md** in repository root
2. **Add export warning to README.md**
3. **Implement download geoblocking** (if hosting models outside GitHub)
4. **Monitor access logs** for sanctioned country attempts
5. **Review collaborators** for foreign national issues

### 7.3 Model Release Checklist

Before releasing trained model weights:

- [ ] **Document Architecture** - Model card with capabilities/limitations
- [ ] **Disclose Training Data** - Sources, biases, privacy considerations
- [ ] **State Intended Use** - Authorized use cases only
- [ ] **Prohibit Misuse** - Clear prohibited use statement
- [ ] **Export Warning** - Include export control notice in model metadata
- [ ] **License Terms** - Export compliance in license agreement
- [ ] **Contact Info** - Compliance contact for questions

---

## 8. Penalties for Non-Compliance

### 8.1 US Penalties (Same as Main Repo)

**Civil Penalties (EAR):**
- Up to **$368,136 per violation** (adjusted annually)
- Enhanced penalties: up to **$1,840,681 or 2x transaction value**

**Criminal Penalties (EAR):**
- Fines up to **$1,000,000 per violation**
- Imprisonment up to **20 years**

**OFAC Violations:**
- Civil: up to **$368,136 per violation** (strict liability)
- Criminal: up to **$1,000,000** and **20 years imprisonment**

**Additional Consequences:**
- Export privilege denial
- Debarment from government contracts
- Seizure of models/code
- Public disclosure
- Reputation damage

---

## 9. AI-Specific Ethical and Legal Considerations

### 9.1 Responsible AI Deployment

Beyond export controls, consider:

**Ethical Use:**
- **Bias and Fairness** - Models may have biases from training data
- **Privacy** - Behavioral models may violate privacy rights
- **Consent** - Ensure subjects consent to monitoring
- **Transparency** - Disclose AI use in security operations

**Legal Compliance:**
- **GDPR** (EU) - Right to explanation, data minimization
- **CCPA** (California) - Consumer privacy rights
- **ECPA** (US) - Electronic communications privacy
- **CFAA** (US) - Computer fraud and abuse

### 9.2 Dual-Use AI Ethics

**The Spectre Map Models are dual-use by design:**

**Legitimate Uses:**
- Authorized penetration testing
- Network security monitoring (with consent)
- Threat hunting in owned infrastructure
- Security research in controlled environments

**Illegitimate Uses:**
- Unauthorized surveillance
- Mass monitoring without warrants
- Cyber espionage
- State repression

**Developer Responsibility:**
- We provide tools, not authorization
- Users must obtain legal permission
- We condemn malicious use
- We support responsible disclosure

---

## 10. Emerging AI Export Controls

### 10.1 US AI Regulations Under Development

**Executive Order 14110 (October 2023):**
- Requires reporting of large AI model training
- Establishes AI safety standards
- May introduce new export controls on foundation models

**Proposed AI Controls:**
- Models trained with >10^26 FLOPS (Spectre Map Models: well below)
- Biological/chemical/nuclear weapon design AI
- Autonomous weapons AI
- Mass surveillance AI

**Spectre Map Models Status:**
- Below large model training thresholds
- Not foundation models
- Narrow task-specific models
- May still fall under general cybersecurity controls

### 10.2 International AI Regulations

**EU AI Act:**
- High-risk AI systems require conformity assessment
- Biometric identification, critical infrastructure, law enforcement
- Spectre Map Models: Likely "high-risk" in EU

**China AI Regulations:**
- Algorithm registration required
- Content moderation and surveillance AI controlled
- Export restrictions on advanced AI

**Future Considerations:**
- International AI export control coordination
- Multilateral AI governance frameworks
- Industry self-regulation standards

---

## 11. Model Cards and Documentation Requirements

### 11.1 Model Card Requirement

Each model should include a **Model Card** documenting:

1. **Model Details** - Architecture, version, developer
2. **Intended Use** - Authorized use cases
3. **Prohibited Uses** - Explicit no-go scenarios
4. **Training Data** - Sources, size, biases
5. **Performance** - Accuracy, precision, recall, F1
6. **Limitations** - Known failure modes, biases
7. **Ethical Considerations** - Privacy, fairness, accountability
8. **Export Controls** - ECCN, prohibited destinations

See [MODEL_CARDS.md](MODEL_CARDS.md) for templates.

### 11.2 Dataset Documentation

Training datasets should include:

1. **Source** - Where data was obtained
2. **License** - Legal terms of use
3. **Privacy** - PII handling, anonymization
4. **Consent** - Subject consent status
5. **Bias** - Known demographic/geographic biases
6. **Quality** - Data quality metrics

See [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md) for details.

---

## 12. Liability Disclaimer

### 12.1 Developer Disclaimer

**LACKADAISICAL SECURITY EXPRESSLY DISCLAIMS ALL LIABILITY FOR:**
- Unauthorized export or distribution of models
- User non-compliance with export control laws
- Misuse of models for illegal purposes
- Damages arising from export control violations
- Legal fees, penalties, or sanctions incurred by users
- Privacy violations or surveillance abuses
- Bias or discrimination resulting from model use

**Users acknowledge and agree that:**
1. They are solely responsible for export compliance
2. They will not export/distribute to sanctioned countries or persons
3. They will obtain all necessary licenses and authorizations
4. They will use models only for lawful, authorized purposes
5. They will respect privacy and human rights
6. They will document and mitigate bias

### 12.2 Model Use Restrictions

**You MAY use these models for:**
- ✅ Authorized penetration testing (with written permission)
- ✅ Security research in controlled environments
- ✅ Defensive security operations on owned infrastructure
- ✅ Educational purposes in sanctioned institutions
- ✅ Threat hunting with proper authorization

**You MAY NOT use these models for:**
- ❌ Unauthorized network intrusion
- ❌ Mass surveillance without legal authority
- ❌ Cyber espionage or intelligence collection
- ❌ State repression or human rights violations
- ❌ Export to sanctioned countries
- ❌ Transfer to denied parties

---

## 13. Updates and Monitoring

### 13.1 Regulatory Monitoring

**Export control regulations change frequently. Monitor:**

- **BIS Website:** https://www.bis.doc.gov (monthly check)
- **OFAC Sanctions:** https://www.treasury.gov/ofac (weekly updates)
- **Federal Register:** https://www.federalregister.gov (for proposed rules)
- **AI Export Controls:** Emerging regulations (subscribe to alerts)

**Update Schedule:**
- Review this document **quarterly** at minimum
- Update immediately upon new AI export controls
- Rebuild models if regulations change classification

### 13.2 Audit and Recordkeeping

**Maintain records for 5+ years:**
- Model distribution logs (who downloaded what, when)
- Export license applications and approvals
- Denied party screening results
- End-use certifications
- Training data sources and licenses
- Model performance metrics

---

## 14. Contact and Resources

### 14.1 Lackadaisical Security Compliance Contact

**For compliance questions:**
- **Email:** lackadaisicalresearch@pm.me
- **Subject Line:** [EXPORT COMPLIANCE] Your Question
- **XMPP+OTR:** thelackadaisicalone@xmpp.jp

**We are NOT export control attorneys. Consult legal counsel for specific guidance.**

### 14.2 US Government Resources

**Bureau of Industry and Security (BIS):**
- Website: https://www.bis.doc.gov
- Encryption: https://www.bis.doc.gov/encryption
- Helpline: (202) 482-4811
- Email: ERC@bis.doc.gov

**Office of Foreign Assets Control (OFAC):**
- Website: https://www.treasury.gov/ofac
- Sanctions Search: https://sanctionssearch.ofac.treas.gov
- Hotline: (800) 540-6322

**Directorate of Defense Trade Controls (DDTC):**
- Website: https://www.pmddtc.state.gov
- ITAR Guidance: https://www.pmddtc.state.gov/ddtc_public

### 14.3 Legal Resources

**Find an Export Control Attorney:**
- American Bar Association (ABA) - International Law Section
- International Compliance Professionals Association (ICPA)
- Export control law firms (Google "export control attorney [your state]")

---

## 15. Revision History

| Version | Date       | Changes                                  | Author                |
|---------|------------|------------------------------------------|-----------------------|
| 1.0     | 2026-01-28 | Initial release for AI/ML models        | Lackadaisical Security |

---

## ⚠️ FINAL WARNING

**THIS DOCUMENT IS FOR INFORMATIONAL PURPOSES ONLY AND DOES NOT CONSTITUTE LEGAL ADVICE.**

AI/ML export controls are complex, rapidly evolving, and carry severe penalties for violations. **You must consult with a qualified export control attorney** before:
- Distributing trained models internationally
- Sharing model weights with foreign nationals
- Publishing training datasets
- Commercializing models
- Collaborating with international researchers
- Using models in government/military contexts

**Lackadaisical Security assumes no responsibility for your compliance with export control laws. You download, train, and deploy these models at your own legal risk.**

**If you can't afford a lawyer, you can't afford to violate export controls.**

---

### 🔥 **Secured by Lackadaisical Security** 🔥

*"In the realm where neural networks become weapons, where trained weights hold the power to detect or deceive, where algorithms decide the fate of nations in microseconds—only those who understand both the technical art and the legal boundaries shall wield such power responsibly. These models are not toys. They are instruments of digital warfare, crafted with precision, governed by law, and deployed only by those who bear the weight of authorization."*

— **Lackadaisical Security, The Operator** (2025)

**Copyright © 2025-2026 Lackadaisical Security. All rights reserved.**