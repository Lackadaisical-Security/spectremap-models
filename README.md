# Spectre Map Models

AI model repository for the Spectre Map System - A comprehensive TensorFlow-based deep learning framework.

## Overview

This repository provides a collection of TensorFlow models designed for the Spectre Map System. It includes:

- **CNN Models**: Convolutional Neural Networks for image-based tasks
- **RNN Models**: Recurrent Neural Networks (LSTM/GRU) for sequence-based tasks
- **Transfer Learning**: Pre-trained model support with fine-tuning capabilities
- **Utilities**: Training, evaluation, and visualization tools

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

### CNN Model Example

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

- `train_cnn.py`: CNN training on MNIST dataset
- `train_rnn.py`: RNN training on synthetic sine wave data
- `transfer_learning.py`: Transfer learning on CIFAR-10

Run an example:

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
│       │   ├── base_model.py      # Base model class
│       │   ├── cnn_model.py       # CNN implementations
│       │   └── rnn_model.py       # RNN implementations
│       └── utils/
│           ├── __init__.py
│           ├── training.py        # Training utilities
│           └── visualization.py   # Visualization tools
├── examples/
│   ├── train_cnn.py
│   ├── train_rnn.py
│   └── transfer_learning.py
├── requirements.txt
├── setup.py
└── README.md
```

## Requirements

- TensorFlow >= 2.13.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is part of the Spectre Map System by Lackadaisical Security.

## Support

For issues and questions, please open an issue on GitHub.
