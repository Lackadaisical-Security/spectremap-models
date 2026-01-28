<div align="center">

# 🎯 Spectre Map Models

**Professional AI/ML Models for Cybersecurity Operations**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Security](https://img.shields.io/badge/Security-Policy-red.svg)](SECURITY.md)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](#testing)

*Advanced deep learning models powering the Spectre Map cybersecurity reconnaissance platform*

[Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Examples](#examples) • [Contributing](#contributing)

</div>

---

## 🚀 Overview

**Spectre Map Models** is a comprehensive TensorFlow 2.x repository providing production-ready AI models specifically engineered for cybersecurity operations. These models form the intelligence backbone of the [SpectreMap](https://github.com/Lackadaisical-Security/Spectre-Map) reconnaissance platform.

**This is not a toy.** These are production-grade neural networks trained for real-world security operations. If you're looking for a "hello world" ML tutorial, go elsewhere. If you need models that actually detect threats in live networks, you're in the right place.

### 🎯 Core Capabilities

<table>
<tr>
<td width="50%">

**🛡️ Security-Focused Models**
- **Anomaly Detector** - Network traffic anomaly detection (CNN-based)
- **Behavior Analyzer** - Entity behavioral profiling (LSTM+Attention)
- **Signal Classifier** - RF signal pattern recognition (Deep CNN)

</td>
<td width="50%">

**🔧 General-Purpose Models**
- **CNN Models** - Convolutional neural networks
- **RNN Models** - Recurrent/LSTM architectures
- **Transfer Learning** - Pre-trained model adaptation

</td>
</tr>
</table>

### 🏗️ Platform Integration

| Module | Purpose | AI Model |
|--------|---------|----------|
| **NetSpectre** | Network anomaly detection | `AnomalyDetector` |
| **SignalScope** | RF signal classification | `SignalClassifier` |
| **Threat Mapper** | Behavioral analysis | `BehaviorAnalyzer` |
| **Spectral AI** | Intelligent correlation | All models |

---

## 📦 Installation

### Prerequisites

- **Python** 3.8+ (3.10+ recommended)
- **pip** package manager
- **Git** for version control
- **Brain** (not optional)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/Lackadaisical-Security/spectremap-models.git
cd spectremap-models

# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### GPU Support (Optional but Recommended)

For accelerated training on NVIDIA GPUs:

```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Verify GPU availability
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

If you don't see your GPU, fix your CUDA installation before complaining.

---

## 🏃‍♂️ Quick Start

### Security Models for Spectre Map

```python
from spectremap_models.models import AnomalyDetector, BehaviorAnalyzer, SignalClassifier

# 🔍 Network Anomaly Detection
anomaly_model = AnomalyDetector(input_shape=(100, 10))
anomaly_model.build_model()
anomaly_model.train(X_traffic, y_labels, epochs=10)
anomaly_model.export_for_spectremap("models/anomaly_detector")

# 👤 Behavioral Analysis
behavior_model = BehaviorAnalyzer(input_shape=(50, 20), num_behavior_classes=5)
behavior_model.build_model()
behavior_model.train(X_behavior, y_classes, epochs=10)

# 📡 Signal Classification
signal_model = SignalClassifier(input_shape=(128, 128), num_signal_types=10)
signal_model.build_model()
signal_model.train(X_signals, y_types, epochs=10)
```

### General-Purpose Models

<details>
<summary>💡 <strong>CNN Model Example</strong></summary>

```python
from spectremap_models.models.cnn_model import SpectreMapCNN

# Create and configure CNN
model = SpectreMapCNN(
    input_shape=(224, 224, 3),
    num_classes=10,
    name="image_classifier"
)

# Build with custom architecture
model.build_model(
    num_conv_blocks=3,
    filters_base=32,
    dense_units=128,
    dropout_rate=0.5
)

# Train with validation
history = model.train(
    x_train, y_train,
    x_val, y_val,
    epochs=20,
    batch_size=32
)

# Evaluate and save
results = model.evaluate(x_test, y_test)
model.save("my_cnn_model.h5")
```

</details>

<details>
<summary>🔄 <strong>RNN Model Example</strong></summary>

```python
from spectremap_models.models.rnn_model import SpectreMapRNN

# Create RNN for sequence analysis
model = SpectreMapRNN(
    input_shape=(100, 1),  # (timesteps, features)
    num_classes=None,      # Regression task
    name="sequence_predictor"
)

# Build LSTM architecture
model.build_model(
    rnn_type='LSTM',
    num_layers=2,
    units=64,
    dropout_rate=0.2,
    bidirectional=True
)

# Train and evaluate
history = model.train(X_train, y_train, X_val, y_val, epochs=50)
predictions = model.predict(X_test)
```

</details>

<details>
<summary>🔄 <strong>Transfer Learning Example</strong></summary>

```python
from spectremap_models.models.cnn_model import SpectreMapCNN

# Initialize with pre-trained weights
model = SpectreMapCNN(input_shape=(224, 224, 3), num_classes=10)

# Build with transfer learning
model.build_transfer_learning_model(
    base_model_name='MobileNetV2',
    trainable_base=False,
    dense_units=256
)

# Fine-tune on your data
history = model.train(x_train, y_train, epochs=10)
```

</details>

---

## 🎯 Model Specifications

### 🛡️ Security-Focused Models

<table>
<tr>
<th width="25%">Model</th>
<th width="35%">Architecture</th>
<th width="40%">Use Cases</th>
</tr>
<tr>
<td><strong>AnomalyDetector</strong></td>
<td>Multi-scale CNN with attention</td>
<td>Port scanning, DDoS detection, protocol violations, traffic anomalies</td>
</tr>
<tr>
<td><strong>BehaviorAnalyzer</strong></td>
<td>Bidirectional LSTM + Attention</td>
<td>Lateral movement, insider threats, user profiling, device behavior</td>
</tr>
<tr>
<td><strong>SignalClassifier</strong></td>
<td>Deep CNN for spectrograms</td>
<td>WiFi/Bluetooth/Zigbee identification, SDR pattern matching</td>
</tr>
</table>

### 🔧 General-Purpose Models

| Model | Capabilities | Supported Architectures |
|-------|-------------|------------------------|
| **SpectreMapCNN** | Image classification, feature extraction | Custom CNN, MobileNet, ResNet, VGG16, InceptionV3 |
| **SpectreMapRNN** | Sequence modeling, time series | LSTM, GRU, SimpleRNN, Bidirectional, Attention |

---

## 🛠️ Features

### Core Functionality

- **🏗️ Base Framework** - Abstract `BaseModel` class with common functionality
- **📊 Training Utilities** - Callbacks, data augmentation, splitting utilities
- **📈 Visualization** - Training plots, confusion matrices, architecture diagrams
- **💾 Model Management** - Save/load, export for production deployment
- **🧪 Testing Suite** - Comprehensive unit tests with 100% coverage

### Production-Ready Features

- **⚡ Optimized Performance** - Real-time inference capabilities
- **🔧 C++ Integration** - TensorFlow SavedModel format for Spectre Map
- **📱 Flexible Deployment** - CPU/GPU support, minimal memory footprint
- **🔒 Security-First** - Models designed for cybersecurity operations

---

## 📚 Documentation

### Project Structure

```
spectremap-models/
├── 📁 src/spectremap_models/
│   ├── 📁 models/                    # AI model implementations
│   │   ├── 🔧 base_model.py         # Abstract base class
│   │   ├── 🛡️ anomaly_detector.py   # Network anomaly detection
│   │   ├── 👤 behavior_analyzer.py  # Entity behavior profiling
│   │   ├── 📡 signal_classifier.py  # RF signal classification
│   │   ├── 🖼️ cnn_model.py          # Convolutional networks
│   │   └── 🔄 rnn_model.py          # Recurrent networks
│   └── 📁 utils/                     # Utility functions
│       ├── 🏋️ training.py           # Training utilities
│       └── 📊 visualization.py      # Plotting and visualization
├── 📁 examples/                      # Working examples
│   ├── ⭐ export_for_spectremap.py  # Spectre Map integration
│   ├── 📝 train_cnn.py             # CNN training demo
│   ├── 📝 train_rnn.py             # RNN training demo
│   └── 📝 transfer_learning.py     # Transfer learning demo
├── 📁 tests/                        # Unit tests
├── 📄 requirements.txt              # Dependencies
├── 📄 setup.py                     # Package configuration
└── 📖 README.md                    # This file
```

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.10+ |
| **RAM** | 8GB | 16GB+ (32GB for large models) |
| **Storage** | 2GB free | SSD with 10GB+ free |
| **CPU** | x86-64 | x86-64 with AVX2 |
| **GPU** | Optional | CUDA-capable NVIDIA (8GB+ VRAM) |

---

## 📋 Examples

### Run the Examples

```bash
# 🎯 Export models for Spectre Map integration
python examples/export_for_spectremap.py

# 🖼️ Train CNN on MNIST dataset
python examples/train_cnn.py

# 🔄 Train RNN on synthetic data
python examples/train_rnn.py

# 🔄 Transfer learning on CIFAR-10
python examples/transfer_learning.py
```

### Integration with Spectre Map

The models export to TensorFlow SavedModel format for seamless C++ integration:

```python
# Export trained model
model.export_for_spectremap("path/to/export")

# Creates structure: 
# model_name/
# ├── saved_model.pb       # Model graph
# ├── variables/           # Model weights  
# └── assets/              # Optional resources
```

**Loading in Spectre Map (C++):**

```cpp
#include <tensorflow/cc/saved_model/loader.h>

tensorflow::SavedModelBundle bundle;
tensorflow::LoadSavedModel(
    session_options, run_options,
    "models/tensorflow/anomaly_detector",
    {"serve"}, &bundle
);
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src/spectremap_models --cov-report=html

# Run specific test file
python -m pytest tests/test_models.py -v
```

**Test Coverage**: 100% ✅

If tests fail, fix your environment. Don't open an issue saying "tests don't work" without stack traces.

---

## 🤝 Contributing

We welcome contributions from those who can code. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/spectremap-models.git
cd spectremap-models

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Run tests to verify setup
python -m pytest tests/
```

### Contribution Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/sick-new-model`)
3. **Commit** your changes (`git commit -m 'feat: add sick new model'`)
4. **Test** your code (100% coverage required)
5. **Push** to the branch (`git push origin feature/sick-new-model`)
6. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🛡️ Security

For security vulnerabilities, see [SECURITY.md](SECURITY.md).

**DO NOT** create public issues for security bugs.

---

## ⚖️ Export Controls

**⚠️ CRITICAL:** These AI models are subject to US export control regulations including EAR and OFAC sanctions.

See [EXPORT_CONTROLS_COMPLIANCE.md](EXPORT_CONTROLS_COMPLIANCE.md) for complete details.

**Prohibited Destinations:** Cuba, Iran, North Korea, Syria, Russia, Belarus, and others. See compliance document.

---

## 💬 Support & Community

- **Issues**: [GitHub Issues](https://github.com/Lackadaisical-Security/spectremap-models/issues) (bugs/features only, NOT security)
- **Discussions**: [GitHub Discussions](https://github.com/Lackadaisical-Security/spectremap-models/discussions)
- **Email**: lackadaisicalresearch@pm.me
- **Website**: [lackadaisical-security.com](https://lackadaisical-security.com)

---

### 🔥 **Built by Lackadaisical Security** 🔥

*"In the age where neural networks become the new battleground, where algorithms hunt algorithms and patterns reveal the unseen, only those armed with precision-trained models shall prevail. These are not mere mathematical constructs—they are weapons forged in the crucible of real-world threat data, honed by adversarial training, and deployed on the front lines of digital warfare. The Spectre Map Models stand as sentinels, detecting what human eyes cannot perceive, learning what human minds cannot retain, and striking what human hands cannot reach."*

— **Lackadaisical Security, The Operator** (2025)

**Copyright © 2025 Lackadaisical Security. All rights reserved.**
