"""
Example: Building and exporting models for Spectre Map integration.

This script demonstrates how to create, train, and export TensorFlow models
in the format expected by the Spectre Map platform.
"""

import numpy as np
import tensorflow as tf
from spectremap_models.models.anomaly_detector import AnomalyDetector
from spectremap_models.models.behavior_analyzer import BehaviorAnalyzer
from spectremap_models.models.signal_classifier import SignalClassifier
import os

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


def generate_synthetic_traffic_data(num_samples=5000, timesteps=100, features=10):
    """Generate synthetic network traffic data for anomaly detection."""
    X_normal = []
    X_anomaly = []
    
    # Normal traffic
    for _ in range(int(num_samples * 0.7)):
        # Regular pattern with small variations
        pattern = np.random.normal(0.5, 0.1, (timesteps, features))
        X_normal.append(pattern)
    
    # Anomalous traffic
    for _ in range(int(num_samples * 0.3)):
        # Irregular spikes and patterns
        pattern = np.random.normal(0.5, 0.3, (timesteps, features))
        pattern[np.random.randint(0, timesteps, 10)] *= 3  # Add spikes
        X_anomaly.append(pattern)
    
    X = np.array(X_normal + X_anomaly).astype('float32')
    y = np.array([0] * len(X_normal) + [1] * len(X_anomaly))
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def generate_synthetic_behavior_data(num_samples=3000, timesteps=50, features=20):
    """Generate synthetic entity behavior data."""
    X = []
    y = []
    
    # Define 5 behavior types
    behavior_patterns = {
        0: lambda: np.random.normal(0.3, 0.1, (timesteps, features)),  # Normal user
        1: lambda: np.random.normal(0.7, 0.15, (timesteps, features)),  # Power user
        2: lambda: np.random.normal(0.5, 0.3, (timesteps, features)),  # Irregular
        3: lambda: np.random.normal(0.2, 0.05, (timesteps, features)),  # Minimal activity
        4: lambda: np.random.uniform(0, 1, (timesteps, features)),  # Suspicious/random
    }
    
    for behavior_type in range(5):
        for _ in range(num_samples // 5):
            pattern = behavior_patterns[behavior_type]()
            X.append(pattern)
            y.append(behavior_type)
    
    X = np.array(X).astype('float32')
    y = np.array(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def generate_synthetic_signal_data(num_samples=2000, height=128, width=128):
    """Generate synthetic RF signal spectrograms."""
    X = []
    y = []
    
    # 10 signal types (WiFi, BLE, Zigbee, etc.)
    for signal_type in range(10):
        for _ in range(num_samples // 10):
            # Generate spectrogram-like patterns
            spectrogram = np.random.normal(0.5, 0.2, (height, width))
            
            # Add signal-specific patterns
            center_freq = (signal_type + 1) * (height // 12)
            spectrogram[center_freq-5:center_freq+5, :] += 0.3
            
            X.append(spectrogram)
            y.append(signal_type)
    
    X = np.array(X).astype('float32')
    y = np.array(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def main():
    print("=== Building Models for Spectre Map Integration ===\n")
    
    # Create export directory
    export_dir = "spectremap_exported_models"
    os.makedirs(export_dir, exist_ok=True)
    
    # 1. Anomaly Detector
    print("1. Building Anomaly Detector...")
    X_traffic, y_traffic = generate_synthetic_traffic_data()
    
    anomaly_model = AnomalyDetector(input_shape=(100, 10))
    anomaly_model.build_model(
        conv_filters=[64, 128],
        dense_units=128,
        learning_rate=0.001
    )
    
    print("   Training...")
    anomaly_model.train(
        X_traffic, y_traffic,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    print("   Exporting for Spectre Map...")
    anomaly_model.export_for_spectremap(f"{export_dir}/anomaly_detector")
    
    # 2. Behavior Analyzer
    print("\n2. Building Behavior Analyzer...")
    X_behavior, y_behavior = generate_synthetic_behavior_data()
    
    behavior_model = BehaviorAnalyzer(
        input_shape=(50, 20),
        num_behavior_classes=5
    )
    behavior_model.build_model(
        lstm_units=[128, 64],
        dense_units=64,
        learning_rate=0.001
    )
    
    print("   Training...")
    behavior_model.train(
        X_behavior, y_behavior,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    print("   Exporting for Spectre Map...")
    behavior_model.export_for_spectremap(f"{export_dir}/behavior_lstm")
    
    # 3. Signal Classifier
    print("\n3. Building Signal Classifier...")
    X_signals, y_signals = generate_synthetic_signal_data()
    
    signal_model = SignalClassifier(
        input_shape=(128, 128),
        num_signal_types=10
    )
    signal_model.build_model(
        conv_blocks=3,
        filters_base=32,
        dense_units=256,
        learning_rate=0.001
    )
    
    print("   Training...")
    signal_model.train(
        X_signals, y_signals,
        epochs=5,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    print("   Exporting for Spectre Map...")
    signal_model.export_for_spectremap(f"{export_dir}/signal_classifier")
    
    print(f"\n✅ All models exported to: {export_dir}/")
    print("\nIntegration Instructions:")
    print("1. Copy the exported models to Spectre Map's models/tensorflow/ directory")
    print("2. Models are in SavedModel format (.pb) ready for TensorFlow C++ API")
    print("3. Load in Spectre Map using: tf.saved_model.load()")
    
    print("\nExported Models:")
    for model_name in ["anomaly_detector", "behavior_lstm", "signal_classifier"]:
        model_path = f"{export_dir}/{model_name}"
        if os.path.exists(model_path):
            size_mb = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(model_path)
                for filename in filenames
            ) / (1024 * 1024)
            print(f"  - {model_name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
