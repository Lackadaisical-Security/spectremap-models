"""
Example: Training an RNN model for sequence prediction.
"""

import numpy as np
import tensorflow as tf
from spectremap_models.models.rnn_model import SpectreMapRNN
from spectremap_models.utils.training import create_callbacks, split_data
from spectremap_models.utils.visualization import plot_training_history, plot_predictions

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def generate_sine_wave_data(num_samples=10000, timesteps=50, features=1):
    """Generate synthetic sine wave data for sequence prediction."""
    X = []
    y = []
    
    for _ in range(num_samples):
        # Random frequency and phase
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Generate sequence
        t = np.linspace(0, 4 * np.pi, timesteps + 1)
        sequence = np.sin(freq * t + phase)
        
        # Input is all but last timestep, output is last value
        X.append(sequence[:-1].reshape(-1, features))
        y.append(sequence[-1])
    
    return np.array(X), np.array(y)


def main():
    print("=== Spectre Map RNN Example ===\n")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = generate_sine_wave_data(num_samples=5000, timesteps=50)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}\n")
    
    # Create model
    print("Creating RNN model...")
    model = SpectreMapRNN(
        input_shape=(50, 1),
        num_classes=None,  # Regression task
        name="sine_wave_rnn"
    )
    
    # Build model
    model.build_model(
        rnn_type='LSTM',
        num_layers=2,
        units=64,
        dropout_rate=0.2,
        recurrent_dropout=0.2,
        learning_rate=0.001
    )
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(
        model_name="sine_wave_rnn",
        early_stopping_patience=10,
        reduce_lr_patience=5
    )
    
    # Train model
    print("\nTraining model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=64,
        callbacks=callbacks
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(X_test, y_test)
    print(f"Test results: {results}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    
    # Plot results
    print("\nPlotting results...")
    plot_training_history(history, save_path="rnn_training_history.png")
    plot_predictions(y_test, predictions.flatten(), save_path="rnn_predictions.png")
    
    # Save model
    print("\nSaving model...")
    model.save("saved_models/sine_wave_rnn")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
