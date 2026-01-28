# Code of Conduct

## Philosophy

Spectre Map Models is built on a foundation of **technical excellence, meritocracy, and logic**. This project operates under old-school hacker ethics from the leet era: your code speaks louder than anything else. If you can contribute quality models, prove your ML/AI expertise, and operate with integrity, you're in.

**No politics. No bullshit. Just neural networks.**

## Core Principles

### 1. Merit is King
- **Your contributions are judged by technical merit only** - model accuracy, training efficiency, clean code, reproducible results
- Skill level doesn't matter if you're willing to learn and improve
- Show your work. Explain your architecture. Defend your hyperparameters with data
- If your model performs worse than baseline, explain why or improve it

### 2. Technical Competence Over Everything
- Know what you're talking about or shut up
- Understand the math behind your models
- Research before you ask questions - read the papers, check the docs
- "My model doesn't train" is not a bug report - show loss curves, gradients, logs
- If you claim SOTA performance, provide benchmarks and reproducibility

### 3. Intellectual Honesty
- Don't plagiarize models or training code. Cite your sources
- If you don't know something, say so - nobody expects you to know every architecture
- Admit when your model is overfitting or biased
- Document your experiments honestly - include failed attempts and lessons learned
- Cherry-picking metrics is bullshit - report all relevant performance indicators

### 4. Hacker Ethics (Classic)
- **Access to AI should be unlimited and total** - but you better have authorization to use it
- **All models should be open** - but stealing pretrained weights isn't "liberation," it's theft
- **Mistrust blackbox AI – promote explainability** - understand what your models learn
- **You can create art and intelligence on a GPU** - make elegant architectures, not bloated garbage
- **AI can change the world for the better** - build models that actually solve problems

## What We Expect

### Technical Standards
- Write clean, documented, reproducible code
- Follow PEP 8 style (use `black`, `flake8`, `mypy`)
- Provide reproducible training scripts with seeds
- Version control your experiments (not just code - track hyperparameters)
- Security matters - don't leak training data, don't train on PII without consent
- Performance matters - optimize inference speed, don't waste GPU cycles

### Communication Standards
- Be direct and honest - no passive-aggressive nonsense
- Technical criticism is not personal - "your model sucks" means fix the architecture
- If someone's training code is inefficient, explain WHY and HOW to fix it
- Argue about architectures, hyperparameters, and optimization - not personalities
- Keep discussions on-topic and focused on the models

### Collaboration Standards
- Review others' code honestly - don't approve garbage models
- Respond to feedback constructively - defend your approach with ablation studies
- Share knowledge when asked - the community grows when experts teach
- If you promise to train a model, deliver it or say you can't
- Open-source your best work - don't hoard knowledge

## What We Don't Tolerate

### Hard Bans (Instant Removal)
- **Using models for unauthorized surveillance/espionage** - we're not covering your ass in court
- **Training on stolen/unauthorized datasets** - respect data privacy and licenses
- **Doxxing, harassment, or stalking** - this is an AI research project, not a drama club
- **Stealing pretrained weights and claiming them as your own** - plagiarism is for script kiddies
- **Publishing biased models without disclosure** - if your model discriminates, document it
- **Deliberately introducing backdoors into models** - you'll be reported to authorities
- **Exporting models to sanctioned countries** - read EXPORT_CONTROLS_COMPLIANCE.md

### Soft Bans (Warning → Kick)
- Repeatedly submitting models with no performance metrics
- Arguing without data ("I think batch size 128 is better" - cool, show the learning curves)
- Not following contribution guidelines after being told multiple times
- Wasting maintainers' time with questions covered in documentation
- Training models on copyrighted data without permission

### What's NOT a Violation
- Using "offensive" language in technical discussions - we're adults
- Disagreeing strongly with architecture choices - if you have a better one, prove it
- Calling out poorly trained models - that's literally what peer review is for
- Being blunt or direct - we value efficiency over hand-holding
- Memes, jokes, and ML culture references - this is part of the tradition
- Healthy skepticism of SOTA claims - "pics or it didn't happen" applies to benchmarks

## Legal & Ethical Use

### Authorized Use Requirements
This is not negotiable. **You MUST:**
- Obtain **written authorization** before deploying models for surveillance/security
- Comply with all applicable laws (GDPR, CCPA, ECPA, CFAA, export controls)
- Respect data privacy - don't train on PII without consent
- Follow responsible AI practices (fairness, transparency, accountability)
- Disclose model capabilities and limitations to end-users

### Prohibited Activities
You will be banned and potentially reported if you:
- Deploy models for unauthorized surveillance or mass monitoring
- Use models for cyber espionage or state-sponsored hacking
- Train on stolen datasets, leaked data, or unauthorized sources
- Export models to sanctioned countries (Cuba, Iran, North Korea, Syria, Russia, Belarus)
- Violate export control laws (see EXPORT_CONTROLS_COMPLIANCE.md)
- Create intentionally biased/discriminatory models without disclosure
- Use models to violate human rights

**If you get arrested for doing dumb shit with these models, you're on your own.**

## AI/ML-Specific Ethics

### Bias and Fairness
- **Document bias** - If your model performs differently across demographics, disclose it
- **Mitigate harm** - Don't deploy models that discriminate without justification
- **Test across distributions** - Don't just optimize for one population
- **Be transparent** - Users deserve to know when AI is making decisions about them

### Privacy and Consent
- **Training data privacy** - Don't train on private data without consent
- **Inference privacy** - Don't use models to re-identify anonymized individuals
- **Data minimization** - Don't collect more data than necessary
- **Right to explanation** - Provide explanations when models make high-stakes decisions

### Dual-Use Concerns
These models are designed for cybersecurity - they have legitimate and illegitimate uses:

**Legitimate:**
- Authorized penetration testing
- Defensive security operations
- Threat hunting in owned infrastructure
- Security research with permission

**Illegitimate:**
- Unauthorized intrusion detection evasion
- State surveillance without warrants
- Cyber espionage
- Human rights violations

**Use responsibly or don't use at all.**

## Enforcement

### Who Enforces
Project maintainer (Lackadaisical Security) has final say. This is not a democracy.

### How to Report Issues
- **Training bugs**: Open a GitHub issue with full reproduction (code, logs, environment)
- **Security vulnerabilities**: Email lackadaisicalresearch@pm.me (PGP preferred)
- **Model bias/ethical concerns**: Email with evidence and analysis
- **Code of conduct violations**: Email lackadaisicalresearch@pm.me with evidence
- **Illegal use of models**: Report to appropriate law enforcement, cc us if relevant

### Consequences
1. **First offense**: Warning via email/issue comment - fix the problem
2. **Second offense**: Temporary ban (duration depends on severity)
3. **Third offense / Severe violations**: Permanent ban from project
4. **Criminal activity**: Reported to authorities, permanent ban, legal action if applicable

### Appeals
If you think you were banned unfairly, email with a technical/data-driven explanation. If you can't defend your position with evidence, the ban stands.

## Attribution

This Code of Conduct is **NOT** based on Contributor Covenant or any corporate template.

This is based on:
- **Hacker Ethic** (Steven Levy, 1984)
- **ML Research Culture** (NeurIPS, ICML, ICLR community standards)
- **Old-school open source** (Linux kernel, BSD culture)
- **Meritocratic principles** of technical communities
- **Responsible AI guidelines** (Partnership on AI, IEEE Ethics)

## Philosophy: Why This Approach?

Spectre Map Models are **production-grade AI for offensive security**. The stakes are incredibly high:
- Models can enable mass surveillance if misused
- Biased models can lead to discrimination and harm
- Poor training can lead to false positives in live security operations
- Export control violations carry criminal penalties
- Privacy violations can destroy lives

**We need contributors who:**
- Take AI ethics seriously
- Can handle direct technical criticism
- Prioritize reproducibility and transparency
- Understand the legal and ethical implications
- Can defend their models with data, not opinions

If you're looking for a "safe space" where your 50% accuracy model gets praised, this isn't it. If you want to build cutting-edge security AI with people who care about both performance and ethics, welcome aboard.

## Contact

**Maintainer**: Lackadaisical Security  
**Email**: lackadaisicalresearch@pm.me  
**XMPP+OTR**: thelackadaisicalone@xmpp.jp  
**Website**: https://lackadaisical-security.com  
**GitHub**: https://github.com/Lackadaisical-Security

---

**TL;DR**: Be competent. Be honest. Document your models. Don't train on stolen data. Don't deploy without authorization. Performance and ethics both matter.

**Copyright © 2025-2026 Lackadaisical Security. All rights reserved.**