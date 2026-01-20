# Spectre Map Models

**TensorFlow Models for Cybersecurity Reconnaissance & Threat Intelligence**

AI model repository for the **Spectre Map System** - Professional-grade deep learning models for security operations.

## Overview

This repository provides TensorFlow 2.x models specifically designed for the **Spectre Map** platform's AI-powered reconnaissance and threat analysis capabilities. These models power Spectre Map's:

- **NetSpectre Module**: Network anomaly detection and threat pattern recognition
- **SignalScope Module**: RF signal classification and wireless protocol identification  
- **Threat Mapper**: Behavioral analysis and attack path prediction
- **Spectral AI Assistant**: Intelligent threat correlation and analysis

### Security-Focused Models

- **Anomaly Detector**: CNN-based network traffic anomaly detection for identifying port scans, DDoS, and protocol violations
- **Behavior Analyzer**: LSTM-based entity profiling for lateral movement detection and insider threat identification
- **Signal Classifier**: CNN-based RF signal classification for WiFi/BLE/Zigbee/SDR pattern recognition
- **CNN Models**: General-purpose convolutional neural networks for image-based security tasks
- **RNN Models**: Recurrent neural networks for sequential threat data analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install from source

```bash
git clone https://github.com/Lackadaisical-Security/spectremap-models.git
cd spectremap-models
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## Quick Start

### For Spectre Map Integration

```python
from spectremap_models.models.anomaly_detector import AnomalyDetector
from spectremap_models.models.behavior_analyzer import BehaviorAnalyzer
from spectremap_models.models.signal_classifier import SignalClassifier

# 1. Anomaly Detection for Network Traffic
anomaly_model = AnomalyDetector(input_shape=(100, 10))
anomaly_model.build_model()
anomaly_model.train(X_traffic, y_labels, epochs=10)
anomaly_model.export_for_spectremap("models/anomaly_detector")

# 2. Behavioral Analysis for Entity Profiling
behavior_model = BehaviorAnalyzer(input_shape=(50, 20), num_behavior_classes=5)
behavior_model.build_model()
behavior_model.train(X_behavior, y_classes, epochs=10)
behavior_model.export_for_spectremap("models/behavior_lstm")

# 3. Signal Classification for RF Analysis
signal_model = SignalClassifier(input_shape=(128, 128), num_signal_types=10)
signal_model.build_model()
signal_model.train(X_signals, y_types, epochs=10)
signal_model.export_for_spectremap("models/signal_classifier")
```

### General-Purpose Models

#### CNN Model Example

```python
from spectremap_models.models.cnn_model import SpectreMapCNN

# Create a CNN model
model = SpectreMapCNN(
    input_shape=(224, 224, 3),
    num_classes=10,
    name="my_cnn"
)

# Build the model
model.build_model(
    num_conv_blocks=3,
    filters_base=32,
    dense_units=128,
    dropout_rate=0.5
)

# Train the model
history = model.train(
    x_train, y_train,
    x_val, y_val,
    epochs=10,
    batch_size=32
)

# Evaluate
results = model.evaluate(x_test, y_test)

# Save
model.save("my_model.h5")
```

### RNN Model Example

```python
from spectremap_models.models.rnn_model import SpectreMapRNN

# Create an RNN model
model = SpectreMapRNN(
    input_shape=(100, 1),  # (timesteps, features)
    num_classes=None,  # For regression
    name="my_rnn"
)

# Build the model
model.build_model(
    rnn_type='LSTM',
    num_layers=2,
    units=64,
    dropout_rate=0.2
)

# Train and evaluate
history = model.train(X_train, y_train, X_val, y_val, epochs=20)
results = model.evaluate(X_test, y_test)
```

### Transfer Learning Example

```python
from spectremap_models.models.cnn_model import SpectreMapCNN

# Create model with transfer learning
model = SpectreMapCNN(input_shape=(224, 224, 3), num_classes=10)

# Build with pre-trained base
model.build_transfer_learning_model(
    base_model_name='MobileNetV2',
    trainable_base=False,
    dense_units=256
)

# Train
history = model.train(x_train, y_train, epochs=10)
```

## Features

### Security-Focused Models

All models are optimized for cybersecurity operations and integrate seamlessly with Spectre Map:

#### Anomaly Detector
- **Purpose**: Identify network traffic anomalies and attack patterns
- **Architecture**: CNN-based with multi-scale feature extraction
- **Use Cases**: Port scanning, DDoS detection, protocol violations, unusual traffic flows
- **Export Format**: TensorFlow SavedModel (.pb) for C++ integration

#### Behavior Analyzer  
- **Purpose**: Profile entity behavior and detect deviations
- **Architecture**: Bidirectional LSTM with attention mechanism
- **Use Cases**: Lateral movement, insider threats, user profiling, device activity analysis
- **Export Format**: TensorFlow SavedModel (.pb) for C++ integration

#### Signal Classifier
- **Purpose**: Classify RF signals and wireless protocols
- **Architecture**: Deep CNN for spectrogram analysis
- **Use Cases**: WiFi identification, Bluetooth device typing, Zigbee networks, SDR pattern matching
- **Export Format**: TensorFlow SavedModel (.pb) for C++ integration

### Base Model

All models inherit from `BaseModel`, which provides:

- **Training**: Flexible training with validation support
- **Evaluation**: Comprehensive evaluation metrics
- **Prediction**: Easy prediction interface
- **Model Management**: Save/load functionality
- **Summary**: Model architecture visualization

### CNN Models

The `SpectreMapCNN` class offers:

- Customizable convolutional architectures
- Batch normalization and dropout for regularization
- Support for various input shapes and output classes
- Transfer learning with popular pre-trained models:
  - MobileNetV2
  - ResNet50
  - VGG16
  - InceptionV3

### RNN Models

The `SpectreMapRNN` class supports:

- Multiple RNN cell types (LSTM, GRU, SimpleRNN)
- Bidirectional RNN layers
- Attention mechanisms
- Task types:
  - Classification
  - Regression
  - Sequence-to-sequence

### Utilities

#### Training Utilities

- **Callbacks**: Pre-configured callbacks for model checkpointing, early stopping, learning rate reduction, and TensorBoard logging
- **Data Augmentation**: Image augmentation pipelines
- **Data Splitting**: Train/validation/test split utilities

#### Visualization Utilities

- **Training History**: Plot loss and metrics over epochs
- **Confusion Matrix**: Visualize classification results
- **Predictions**: Plot predicted vs actual values
- **Model Architecture**: Generate model architecture diagrams

## Examples

The `examples/` directory contains complete working examples:

- **`export_for_spectremap.py`**: Build and export models for Spectre Map integration ⭐
- `train_cnn.py`: CNN training on MNIST dataset
- `train_rnn.py`: RNN training on synthetic sine wave data
- `transfer_learning.py`: Transfer learning on CIFAR-10

### Run Spectre Map Integration Example

```bash
# Build and export all security models for Spectre Map
python examples/export_for_spectremap.py

# Output: Models exported to spectremap_exported_models/
# - anomaly_detector/  (SavedModel format)
# - behavior_lstm/     (SavedModel format)
# - signal_classifier/ (SavedModel format)
```

Run other examples:

```bash
python examples/train_cnn.py
```

## Project Structure

```
spectremap-models/
├── src/
│   └── spectremap_models/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base_model.py          # Base model class
│       │   ├── anomaly_detector.py    # Network anomaly detection ⭐
│       │   ├── behavior_analyzer.py   # Entity behavior profiling ⭐
│       │   ├── signal_classifier.py   # RF signal classification ⭐
│       │   ├── cnn_model.py           # General CNN models
│       │   └── rnn_model.py           # General RNN/LSTM models
│       └── utils/
│           ├── __init__.py
│           ├── training.py            # Training utilities
│           └── visualization.py       # Visualization tools
├── examples/
│   ├── export_for_spectremap.py       # Spectre Map integration ⭐
│   ├── train_cnn.py
│   ├── train_rnn.py
│   └── transfer_learning.py
├── tests/
│   ├── __init__.py
│   └── test_models.py                 # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

⭐ = Spectre Map-specific files

## Requirements

- TensorFlow >= 2.13.0 (CPU and GPU support)
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0

### Hardware Recommendations

- **CPU**: x86-64 with AVX2 for optimized performance
- **GPU** (Optional): CUDA-capable NVIDIA GPU for faster training
- **RAM**: 8GB minimum, 16GB recommended for large models
- **Storage**: SSD recommended for faster data loading

## Integration with Spectre Map

### Model Export Format

All security models can be exported in TensorFlow SavedModel format (.pb) for integration with Spectre Map's C++ backend:

```python
# Export model for Spectre Map
model.export_for_spectremap("path/to/export")
```

This creates a directory structure compatible with TensorFlow's C++ API:

```
model_name/
├── saved_model.pb          # Model graph definition
├── variables/
│   ├── variables.index
│   └── variables.data-*
└── assets/                 # Optional assets
```

### Loading in Spectre Map (C++)

```cpp
// Load model in Spectre Map's C++ backend
#include <tensorflow/cc/saved_model/loader.h>

tensorflow::SavedModelBundle bundle;
tensorflow::LoadSavedModel(
    session_options,
    run_options,
    "models/tensorflow/anomaly_detector",
    {"serve"},
    &bundle
);
```

### Model Integration Checklist

- [x] Models export in SavedModel format
- [x] Compatible with TensorFlow 2.x C++ API
- [x] Optimized for real-time inference
- [x] Support for CPU and GPU acceleration
- [x] Minimal memory footprint
- [x] Production-grade performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is part of the Spectre Map System by Lackadaisical Security.

## Support

For issues and questions, please open an issue on GitHub.
