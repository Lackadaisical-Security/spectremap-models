"""
Example: Using transfer learning with pre-trained models.
"""

import numpy as np
import tensorflow as tf
from spectremap_models.models.cnn_model import SpectreMapCNN
from spectremap_models.utils.training import create_callbacks, create_data_augmentation

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_sample_data():
    """Load CIFAR-10 dataset as a sample."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    return x_train, y_train, x_test, y_test


def main():
    print("=== Spectre Map Transfer Learning Example ===\n")
    
    # Load data
    print("Loading data...")
    x_train, y_train, x_test, y_test = load_sample_data()
    
    # Use a smaller subset for faster training
    x_train = x_train[:5000]
    y_train = y_train[:5000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}\n")
    
    # Create data augmentation
    print("Creating data augmentation...")
    data_augmentation = create_data_augmentation()
    
    # Create model
    print("Creating transfer learning model...")
    model = SpectreMapCNN(
        input_shape=(32, 32, 3),
        num_classes=10,
        name="cifar10_transfer"
    )
    
    # Build transfer learning model
    print("Building model with MobileNetV2 base...")
    model.build_transfer_learning_model(
        base_model_name='MobileNetV2',
        trainable_base=False,
        dense_units=256,
        dropout_rate=0.5,
        learning_rate=0.001
    )
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(
        model_name="cifar10_transfer",
        early_stopping_patience=5
    )
    
    # Train model
    print("\nTraining model (feature extraction)...")
    history = model.train(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        callbacks=callbacks,
        validation_split=0.2
    )
    
    # Evaluate
    print("\nEvaluating model...")
    results = model.evaluate(x_test, y_test)
    print(f"Test results: {results}")
    
    # Fine-tuning (optional)
    print("\nFine-tuning the model...")
    # Unfreeze the base model
    model.model.layers[1].trainable = True
    
    # Recompile with lower learning rate
    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history_fine = model.train(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2
    )
    
    # Final evaluation
    print("\nFinal evaluation...")
    results = model.evaluate(x_test, y_test)
    print(f"Test results after fine-tuning: {results}")
    
    # Save model
    print("\nSaving model...")
    model.save("saved_models/cifar10_transfer")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
