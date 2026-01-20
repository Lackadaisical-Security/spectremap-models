"""
Example: Training a CNN model for image classification.
"""

import numpy as np
import tensorflow as tf
from spectremap_models.models.cnn_model import SpectreMapCNN
from spectremap_models.utils.training import create_callbacks, split_data
from spectremap_models.utils.visualization import plot_training_history

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_sample_data():
    """Load MNIST dataset as a sample."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Reshape and normalize
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    return x_train, y_train, x_test, y_test


def main():
    print("=== Spectre Map CNN Example ===\n")
    
    # Load data
    print("Loading data...")
    x_train, y_train, x_test, y_test = load_sample_data()
    
    # Split into train and validation
    x_train, x_val, _, y_train, y_val, _ = split_data(
        x_train, y_train, train_ratio=0.8, val_ratio=0.2
    )
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}\n")
    
    # Create model
    print("Creating CNN model...")
    model = SpectreMapCNN(
        input_shape=(28, 28, 1),
        num_classes=10,
        name="mnist_cnn"
    )
    
    # Build model
    model.build_model(
        num_conv_blocks=2,
        filters_base=32,
        dense_units=128,
        dropout_rate=0.5,
        learning_rate=0.001
    )
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(
        model_name="mnist_cnn",
        early_stopping_patience=5,
        reduce_lr_patience=3
    )
    
    # Train model
    print("\nTraining model...")
    history = model.train(
        x_train, y_train,
        x_val, y_val,
        epochs=10,
        batch_size=128,
        callbacks=callbacks
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(x_test, y_test)
    print(f"Test results: {results}")
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, save_path="cnn_training_history.png")
    
    # Save model
    print("\nSaving model...")
    model.save("saved_models/mnist_cnn")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
