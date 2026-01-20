# Contributing to Spectre Map Models

Thank you for your interest in contributing to **Spectre Map Models**! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Release Process](#release-process)

## 📜 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **Git** for version control
- **GitHub account** for contributions
- Basic understanding of **TensorFlow** and **machine learning**

### First-Time Setup

1. **Fork** the repository on GitHub
2. **Clone** your fork locally: 
   ```bash
   git clone https://github.com/YOUR_USERNAME/spectremap-models.git
   cd spectremap-models
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/Lackadaisical-Security/spectremap-models.git
   ```

## 🛠️ Development Setup

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

The development setup includes:

- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **black**: Code formatting
- **flake8**:  Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks
- **sphinx**: Documentation generation

### Verify Setup

```bash
# Run tests to verify everything works
python -m pytest tests/ -v

# Check code formatting
black --check src/ tests/

# Run linting
flake8 src/ tests/

# Type checking
mypy src/
```

## 📝 Contribution Guidelines

### Types of Contributions

We welcome various types of contributions: 

- 🐛 **Bug fixes**
- ✨ **New features**
- 📚 **Documentation improvements**
- 🧪 **Tests and test coverage**
- 🎨 **Code quality improvements**
- 📦 **New model implementations**
- 🔧 **Performance optimizations**

### Contribution Workflow

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** for new features or major changes
3. **Fork and branch** from the main branch
4. **Implement your changes** following our standards
5. **Add tests** for new functionality
6. **Update documentation** as needed
7. **Submit a pull request** with clear description

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows our [coding standards](#coding-standards)
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] Pre-commit hooks pass
- [ ] Branch is up to date with main

### Pull Request Template

```markdown
## Description
Brief description of changes made. 

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainers
3. **Feedback incorporation** if needed
4. **Final approval** and merge

## 🐛 Issue Guidelines

### Reporting Bugs

Use the bug report template and include:

- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (Python version, OS, etc.)
- **Error messages** or logs
- **Minimal code example** if applicable

### Feature Requests

Use the feature request template and include:

- **Clear description** of the proposed feature
- **Use case** and motivation
- **Proposed implementation** (if you have ideas)
- **Alternatives considered**

### Issue Labels

We use labels to organize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Community help needed
- `security`: Security-related issues

## 💻 Coding Standards

### Code Style

We follow **PEP 8** with some modifications:

```python
# Use black for formatting
black src/ tests/

# Line length:  88 characters (black default)
# Import organization: isort compatible
# Docstrings:  Google style
```

### Code Quality Tools

- **Black**:  Automatic code formatting
- **Flake8**: Linting and style checking
- **MyPy**: Static type checking
- **Pre-commit**:  Automated checks before commits

### Best Practices

#### General Guidelines
- Write clear, readable code
- Use descriptive variable and function names
- Follow the principle of least surprise
- Keep functions and classes focused and small

#### ML-Specific Guidelines
- Document model architectures clearly
- Include proper input/output shape documentation
- Add type hints for tensor operations
- Use TensorFlow best practices

#### Example Code Style

```python
from typing import Optional, Tuple
import tensorflow as tf
import numpy as np

class ExampleModel:
    """Example model following our coding standards. 
    
    Args:
        input_shape: Shape of input tensors
        num_classes: Number of output classes
        name: Model name for identification
    """
    
    def __init__(
        self,
        input_shape:  Tuple[int, ... ],
        num_classes: int,
        name: str = "example_model"
    ) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = name
        self.model:  Optional[tf.keras. Model] = None
    
    def build_model(self, **kwargs) -> tf.keras.Model:
        """Build the model architecture. 
        
        Returns:
            Compiled TensorFlow model
        """
        # Implementation here
        pass
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── __init__.py
├── test_models. py          # Model tests
├── test_utils.py           # Utility tests
└── fixtures/               # Test fixtures and data
```

### Testing Standards

- **Coverage**: Aim for >90% code coverage
- **Test Types**:  Unit tests, integration tests
- **Naming**: `test_<functionality>_<expected_outcome>`
- **Documentation**: Clear docstrings for test functions

### Example Test

```python
import pytest
import numpy as np
from spectremap_models. models.cnn_model import SpectreMapCNN

class TestSpectreMapCNN:
    """Test suite for SpectreMapCNN model."""
    
    def test_model_initialization_success(self):
        """Test successful model initialization."""
        model = SpectreMapCNN(
            input_shape=(28, 28, 1),
            num_classes=10,
            name="test_cnn"
        )
        assert model. input_shape == (28, 28, 1)
        assert model.num_classes == 10
        assert model.name == "test_cnn"
    
    def test_model_build_creates_valid_architecture(self):
        """Test that model building creates valid architecture."""
        model = SpectreMapCNN(input_shape=(28, 28, 1), num_classes=10)
        built_model = model. build_model()
        
        assert built_model is not None
        assert len(built_model. layers) > 0
        assert built_model.input_shape == (None, 28, 28, 1)
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src/spectremap_models --cov-report=html

# Run specific test file
python -m pytest tests/test_models.py -v

# Run tests matching pattern
python -m pytest -k "test_cnn" -v
```

## 📚 Documentation

### Documentation Types

- **README**: Project overview and quick start
- **API Documentation**: Function/class documentation
- **Examples**: Working code examples
- **Tutorials**: Step-by-step guides

### Docstring Style

We use **Google-style docstrings**:

```python
def train_model(
    x_train: np. ndarray,
    y_train: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32
) -> tf.keras.callbacks.History:
    """Train the model on provided data.
    
    Args:
        x_train: Training input data of shape (samples, ...)
        y_train: Training target data of shape (samples, ...)
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Training history object containing metrics
        
    Raises: 
        ValueError: If input shapes don't match model requirements
        
    Example: 
        >>> model = SpectreMapCNN(input_shape=(28, 28, 1), num_classes=10)
        >>> model.build_model()
        >>> history = model.train(x_train, y_train, epochs=5)
    """
```

### Documentation Building

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## 🚢 Release Process

### Version Management

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR. PATCH**
- **MAJOR**: Breaking changes
- **MINOR**:  New features, backwards compatible
- **PATCH**:  Bug fixes, backwards compatible

### Release Steps

1. **Update version** in `setup.py` and `__init__.py`
2. **Update changelog** with new features and fixes
3. **Create release PR** with version bump
4. **Merge to main** after approval
5. **Tag release** and publish to PyPI
6. **Update documentation** for new version

## 🏆 Recognition

### Contributors

We recognize contributors in:

- **README.md**: Contributors section
- **CHANGELOG.md**: Release notes
- **GitHub**:  Contributor graphs and statistics

### Types of Recognition

- **First-time contributors**: Special mention
- **Regular contributors**: Maintainer status consideration
- **Significant contributions**: Feature attribution

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Email**: [lackadaisicalresearch@pm.me](mailto:lackadaisicalresearch@pm.me)

### Resources

- **TensorFlow Documentation**: [tensorflow.org](https://tensorflow.org)
- **Python Style Guide**: [PEP 8](https://pep8.org)
- **Git Workflow**: [GitHub Flow](https://guides.github.com/introduction/flow/)

---

## 🙏 Thank You

Thank you for contributing to **Spectre Map Models**! Your contributions help make cybersecurity AI more accessible and effective.

**Happy Contributing!** 🎉
