# Contributing to Spectre Map Models

So you want to contribute AI/ML models for cybersecurity? Good. We need people who can train neural networks, not people who can talk about training neural networks.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Model Development Guidelines](#model-development-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Security & Ethics](#security--ethics)
- [License](#license)

## Code of Conduct

Read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). TL;DR: Be competent, be honest, respect data privacy, don't export to sanctioned countries. Your model performance speaks louder than anything else.

## Getting Started

### Prerequisites

Before you waste anyone's time, make sure you have:

* **Python 3.8+** (3.10+ recommended)
* **TensorFlow 2.13+** or **PyTorch 2.0+** (depending on what you're building)
* **GPU** (optional but highly recommended - training on CPU is pain)
* **Understanding of ML fundamentals** (if you don't know what overfitting is, learn first)
* **Git** (obviously)
* **Brain** (not optional)

### Setting Up Development Environment

#### 1. Fork the Repository
```bash
# Fork via GitHub UI, then clone your fork
git clone https://github.com/YOUR_USERNAME/spectremap-models.git
cd spectremap-models
```

#### 2. Add Upstream Remote
```bash
git remote add upstream https://github.com/Lackadaisical-Security/spectremap-models.git
```

#### 3. Create Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

#### 4. Install Dependencies
```bash
# Install in development mode
pip install -e ".[dev]"

# Or just dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

#### 5. Verify Installation
```bash
# Run tests
python -m pytest tests/ -v

# Check GPU availability
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

If tests fail, fix your environment before opening an issue.

## Development Process

### Branching Strategy

* `main` - Stable production releases (don't touch unless you're a maintainer)
* `develop` - Integration branch for models
* `model/your-model-name` - Your model development branch
* `bugfix/bug-description` - Bug fix branches
* `data/dataset-name` - New training data branches

### Workflow

#### 1. Create a Model Branch
```bash
git checkout -b model/sick-new-detector develop
```

#### 2. Develop Your Model
* Design architecture (document your choices)
* Prepare training data (ethical sourcing, proper licensing)
* Train model (reproducible, with seeds)
* Evaluate performance (multiple metrics, not just accuracy)
* Optimize inference (production deployment matters)
* Document everything (see [Documentation](#documentation))

#### 3. Commit Changes
```bash
git add .
git commit -m "feat(model): add sick new detector for XYZ threats"
```

#### 4. Keep Your Branch Updated
```bash
git fetch upstream
git rebase upstream/develop
```

#### 5. Push Changes
```bash
git push origin model/sick-new-detector
```

#### 6. Create Pull Request
* Open PR against `develop` branch
* Fill out the template completely (include metrics, plots, ablation studies)
* Link related issues
* Wait for review (maintainers will check performance, not just code)

### Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
* `feat` - New model or feature
* `fix` - Bug fix in training/inference code
* `docs` - Documentation only
* `style` - Code formatting (black, flake8)
* `refactor` - Code refactoring
* `perf` - Performance improvements (faster training/inference)
* `test` - Tests
* `data` - Training data changes
* `chore` - Build/tooling changes

**Examples:**
```
feat(model): add LSTM-based behavior analyzer

Implemented bidirectional LSTM with attention mechanism
for entity behavior profiling. Achieves 94.2% accuracy on
synthetic behavioral dataset.

- Architecture: 2-layer BiLSTM + Attention
- Training time: 45 min on V100
- Inference: 12ms per sample
- Model size: 0.7 MB

Closes #42
```

```
perf(training): optimize data loading pipeline

Replaced eager data loading with tf.data.Dataset pipeline.
Reduces training time by 3.5x on ImageNet subset.

Before: 2.4 hours/epoch
After: 41 minutes/epoch

Benchmark results in docs/benchmarks/data_loading.md
```

```
data(anomaly): add synthetic network traffic dataset

Generated 1M synthetic network flows for anomaly detection.
Includes benign traffic + 7 attack types (port scan, DDoS, etc.)

- Source: Synthetic (no PII, no privacy concerns)
- License: MIT (safe for open source)
- Format: CSV with pcap metadata
- Size: 2.3 GB compressed

See data/synthetic_traffic/README.md for details
```

## Model Development Guidelines

### Architecture Design

**Before you code anything:**

1. **Define the problem** - What are you detecting? What's the input/output?
2. **Survey literature** - What architectures work for similar problems?
3. **Start simple** - Baseline model first, then iterate
4. **Justify complexity** - Every layer should have a reason to exist
5. **Consider deployment** - Can this run in production? (latency, memory, GPU requirements)

**Good Architecture:**
```python
# Clear, documented, justified
class AnomalyDetector(BaseModel):
    """
    CNN-based network traffic anomaly detector.
    
    Architecture rationale:
    - Conv layers: Extract local patterns from traffic features
    - Global pooling: Aggregate patterns across time
    - Dense layers: Final classification
    
    Performance: 96.3% accuracy, 8ms inference on CPU
    """
    def build_model(self):
        # Simple, clean, commented
        pass
```

**Bad Architecture:**
```python
# Random layers thrown together
class MyModel:
    def build(self):
        # 47 layers for binary classification (why???)
        # No documentation
        # 2GB model size for 10KB input
        # Overfits on 100 samples
        pass
```

### Training Guidelines

**Reproducibility is MANDATORY:**

```python
# Set seeds for reproducibility
import numpy as np
import tensorflow as tf
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
```

**Training best practices:**

1. **Train/Val/Test Split** - 70/15/15 or 80/10/10, NO DATA LEAKAGE
2. **Cross-Validation** - For small datasets, use k-fold CV
3. **Early Stopping** - Don't overfit (patience=10 epochs is reasonable)
4. **Learning Rate Scheduling** - Reduce on plateau or cosine annealing
5. **Data Augmentation** - If applicable (be careful with security data)
6. **Regularization** - Dropout, L2, etc. (justify your choices)
7. **Batch Size** - Document your choice (and GPU memory constraints)

**Log everything:**
```python
# TensorBoard, Weights & Biases, MLflow, whatever
# We need to see:
# - Loss curves (train + val)
# - Metrics over time
# - Hyperparameters
# - System info (GPU, batch size, etc.)
```

### Performance Metrics

**DON'T just report accuracy.** Security models need comprehensive metrics:

**For Classification:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC, PR-AUC
- False Positive Rate (critical for security!)
- False Negative Rate (missed attacks are bad!)

**For Anomaly Detection:**
- Precision, Recall, F1
- False Alarm Rate
- Detection Rate
- Time to Detection

**For All Models:**
- Inference time (ms per sample)
- Model size (MB)
- GPU memory usage
- Throughput (samples/sec)

**Example metrics report:**
```python
"""
Performance Metrics:
- Accuracy: 96.3%
- Precision: 94.1% (attack detection)
- Recall: 92.8% (don't miss attacks)
- F1-Score: 93.4%
- False Positive Rate: 2.1% (acceptable for production)
- Inference Time: 8ms (CPU), 2ms (GPU)
- Model Size: 0.7 MB (deployable)
- Throughput: 5000 samples/sec (real-time capable)
"""
```

### Dataset Requirements

**Training data MUST be:**

1. **Legally obtained** - No stolen datasets, no scraped private data
2. **Properly licensed** - MIT, CC-BY, or compatible with our license
3. **Ethically sourced** - No PII without consent, respect privacy
4. **Documented** - See [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md)
5. **Reproducible** - Include generation scripts for synthetic data

**Prohibited data sources:**
- ❌ Stolen/leaked datasets
- ❌ PII without consent
- ❌ Copyrighted data without permission
- ❌ Data from sanctioned countries (export control risk)
- ❌ Classified or confidential data

**Acceptable data sources:**
- ✅ Synthetic data (generated, no privacy issues)
- ✅ Public datasets with permissive licenses
- ✅ Your own captured data (with proper authorization)
- ✅ Anonymized/sanitized data (verified privacy-preserving)

## Pull Request Process

### Before Submitting

Check this list or your PR will be rejected:

- [ ] Code follows PEP 8 (run `black`, `flake8`, `mypy`)
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Model achieves reasonable performance (>baseline)
- [ ] Training is reproducible (seeds set, documented)
- [ ] Metrics reported honestly (all of them, not cherry-picked)
- [ ] Model card created (see [MODEL_CARDS.md](MODEL_CARDS.md))
- [ ] Dataset documented (see [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md))
- [ ] Export compliance verified (no prohibited data/destinations)
- [ ] No merge conflicts with develop
- [ ] You actually tested it on unseen data

### PR Checklist

Your PR better include:

1. **Clear title** - "Added model" is not a title
2. **Description**:
   - What problem does this model solve?
   - What architecture did you use? (and WHY?)
   - What performance did you achieve?
   - How did you train it? (hyperparameters, dataset, duration)
   - What are the limitations?
3. **Metrics** - Complete performance report (see above)
4. **Plots** - Loss curves, confusion matrices, ROC curves
5. **Model Card** - Filled out completely
6. **Dataset Documentation** - If using new data
7. **Ablation Studies** (optional but impressive) - What happens if you remove X layer?
8. **Comparison to Baseline** - How much better is this than naive approach?
9. **Deployment Considerations** - Inference time, model size, hardware requirements

### Review Process

1. **Automated Checks** - CI/CD must pass (linting, tests)
2. **Performance Review** - Maintainer will verify metrics
3. **Code Review** - Clean, documented, maintainable code
4. **Ethics Review** - Data sourcing, bias, fairness
5. **Security Review** - Model security, adversarial robustness
6. **Export Compliance** - No prohibited data or destinations

### Merge Requirements

* All CI checks passing
* At least one approving review from maintainer
* Performance meets or exceeds baseline
* No unresolved conversations
* Model card and dataset docs complete
* Export compliance verified

## Coding Standards

### Python Style

Follow [PEP 8](https://pep8.org/) enforced by `black` and `flake8`:

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

**Style requirements:**
* **Line length**: 100 characters (black default)
* **Docstrings**: Google-style docstrings for all public functions/classes
* **Type hints**: Use type annotations (mypy should pass)
* **Naming**:
  * Classes: `PascalCase`
  * Functions/methods: `snake_case`
  * Constants: `UPPER_SNAKE_CASE`
  * Private: `_leading_underscore`

### Example

```python
import tensorflow as tf
from typing import Tuple, Optional

class AnomalyDetector:
    """Network traffic anomaly detector using CNN architecture.
    
    This model detects anomalous network behavior by analyzing
    traffic patterns extracted from packet metadata.
    
    Attributes:
        input_shape: Shape of input traffic features (timesteps, features)
        num_classes: Number of anomaly classes (default: binary)
        model: Compiled Keras model
    
    Example:
        >>> detector = AnomalyDetector(input_shape=(100, 10))
        >>> detector.build_model()
        >>> detector.train(X_train, y_train, epochs=10)
        >>> predictions = detector.predict(X_test)
    """
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int = 2):
        """Initialize anomaly detector.
        
        Args:
            input_shape: Shape of input (timesteps, features)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model: Optional[tf.keras.Model] = None
    
    def build_model(self, filters: int = 32, dropout: float = 0.5) -> None:
        """Build CNN architecture.
        
        Args:
            filters: Number of convolutional filters in first layer
            dropout: Dropout rate for regularization
        
        Raises:
            ValueError: If input_shape is invalid
        """
        # Implementation with clear comments
        pass
```

### Security Considerations

AI/ML models have unique security concerns:

* **Model Security**:
  * Guard against adversarial examples
  * Test model robustness to input perturbations
  * Document known failure modes
  * Consider model inversion attacks (can attacker extract training data?)

* **Data Security**:
  * Never log sensitive training data
  * Sanitize data before saving checkpoints
  * Encrypt model weights if they contain sensitive patterns
  * Be aware of membership inference attacks

* **Deployment Security**:
  * Validate all inputs before inference
  * Rate-limit API endpoints
  * Monitor for model abuse
  * Implement access controls

## Testing Guidelines

### Test Structure

```
tests/
├── test_models.py        # Model architecture tests
├── test_training.py      # Training pipeline tests
├── test_data.py          # Data loading/preprocessing tests
├── test_inference.py     # Inference and prediction tests
└── test_utils.py         # Utility function tests
```

### Writing Tests

Use `pytest`:

```python
import pytest
import numpy as np
from spectremap_models.models import AnomalyDetector

class TestAnomalyDetector:
    """Test suite for AnomalyDetector model."""
    
    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return AnomalyDetector(input_shape=(100, 10))
    
    def test_build_model(self, model):
        """Test that model builds successfully."""
        model.build_model()
        assert model.model is not None
        assert len(model.model.layers) > 0
    
    def test_training_reduces_loss(self, model):
        """Test that training actually improves the model."""
        X = np.random.randn(1000, 100, 10)
        y = np.random.randint(0, 2, size=1000)
        
        model.build_model()
        history = model.train(X, y, epochs=5, verbose=0)
        
        # Loss should decrease
        assert history.history['loss'][-1] < history.history['loss'][0]
    
    def test_inference_shape(self, model):
        """Test that predictions have correct shape."""
        X = np.random.randn(10, 100, 10)
        model.build_model()
        predictions = model.predict(X)
        
        assert predictions.shape == (10, 2)  # Binary classification
```

### Test Coverage

* **Minimum**: 80% code coverage
* **Critical Paths**: 95%+ for model inference code
* **New Models**: 85%+ coverage required

Run coverage:
```bash
pytest tests/ --cov=src/spectremap_models --cov-report=html
```

## Documentation

### Model Cards

**Every model MUST have a Model Card.** See [MODEL_CARDS.md](MODEL_CARDS.md) for template.

Required sections:
1. Model Details
2. Intended Use
3. Prohibited Uses
4. Training Data
5. Performance Metrics
6. Limitations
7. Ethical Considerations
8. Export Controls

### Dataset Documentation

**Every dataset MUST be documented.** See [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md) for template.

### Code Documentation

Use Google-style docstrings:

```python
def train_model(X_train, y_train, epochs=10, batch_size=32):
    """Train the neural network model.
    
    Args:
        X_train (np.ndarray): Training features, shape (n_samples, ...)
        y_train (np.ndarray): Training labels, shape (n_samples,)
        epochs (int): Number of training epochs (default: 10)
        batch_size (int): Batch size for training (default: 32)
    
    Returns:
        tf.keras.callbacks.History: Training history with loss/metrics
    
    Raises:
        ValueError: If X_train and y_train have mismatched lengths
        
    Example:
        >>> history = train_model(X_train, y_train, epochs=20)
        >>> print(f"Final loss: {history.history['loss'][-1]}")
    """
    pass
```

## Security & Ethics

### Security Review

Models touching security-sensitive data require:

1. **Adversarial Testing** - Test against adversarial examples
2. **Robustness Analysis** - Evaluate on out-of-distribution data
3. **Privacy Analysis** - Ensure no training data leakage
4. **Bias Testing** - Check for demographic/geographic bias
5. **Export Compliance** - Verify no prohibited data/destinations

### Reporting Vulnerabilities

**DO NOT** create public issues for security vulnerabilities in models.

Email: **lackadaisicalresearch@pm.me**

Include:
- Which model is vulnerable
- Attack vector (adversarial example, model inversion, etc.)
- Proof of concept
- Suggested mitigation

See [SECURITY.md](SECURITY.md) for responsible disclosure process.

### Ethical Checklist

Before submitting:

- [ ] Training data obtained legally and ethically
- [ ] No PII used without consent
- [ ] Bias documented and mitigated where possible
- [ ] Limitations clearly stated
- [ ] Prohibited uses explicitly listed
- [ ] Export controls verified
- [ ] Privacy implications considered

## License

By contributing, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) for details.

You retain copyright to your contributions, but you grant us a perpetual license to use them.

---

### 🔥 **Built by Lackadaisical Security** 🔥

*"Neural networks are not magic. They are mathematics, optimized through gradient descent, validated through rigorous testing, and deployed with full understanding of their capabilities and limitations. Show us the math. Show us the data. Show us the results. Everything else is noise."*

**Copyright © 2025-2026 Lackadaisical Security. All rights reserved.**