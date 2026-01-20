"""
Training utilities for Spectre Map models.
"""

import tensorflow as tf
from tensorflow import keras
import os
from typing import Optional, List


def create_callbacks(
    model_name: str,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    save_best_only: bool = True
) -> List[keras.callbacks.Callback]:
    """
    Create a standard set of training callbacks.
    
    Args:
        model_name: Name of the model
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory for TensorBoard logs
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        save_best_only: Whether to save only the best model
        
    Returns:
        List of Keras callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_best.h5"),
            monitor='val_loss',
            save_best_only=save_best_only,
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_dir, model_name),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    
    return callbacks


class CustomProgressCallback(keras.callbacks.Callback):
    """Custom callback for tracking training progress."""
    
    def __init__(self, print_every: int = 1):
        """
        Initialize the callback.
        
        Args:
            print_every: Print progress every N epochs
        """
        super().__init__()
        self.print_every = print_every
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        if (epoch + 1) % self.print_every == 0:
            logs = logs or {}
            print(f"\nEpoch {epoch + 1} - ", end="")
            for key, value in logs.items():
                print(f"{key}: {value:.4f} ", end="")
            print()


def create_data_augmentation(
    rotation_range: int = 20,
    width_shift_range: float = 0.2,
    height_shift_range: float = 0.2,
    horizontal_flip: bool = True,
    zoom_range: float = 0.2
) -> keras.Sequential:
    """
    Create a data augmentation pipeline for images.
    
    Args:
        rotation_range: Degree range for random rotations (e.g., 20 for ±20 degrees)
        width_shift_range: Fraction of total width for horizontal shifts
        height_shift_range: Fraction of total height for vertical shifts
        horizontal_flip: Whether to randomly flip images horizontally
        zoom_range: Range for random zoom
        
    Returns:
        Sequential model for data augmentation
    """
    # Convert rotation from degrees to fraction for Keras 3
    rotation_fraction = rotation_range / 360.0
    
    augmentation = keras.Sequential([
        keras.layers.RandomRotation(rotation_fraction),
        keras.layers.RandomTranslation(height_shift_range, width_shift_range),
        keras.layers.RandomZoom(zoom_range),
    ], name="data_augmentation")
    
    if horizontal_flip:
        augmentation.add(keras.layers.RandomFlip("horizontal"))
    
    return augmentation


def split_data(X, y, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: Optional[int] = 42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Input data
        y: Labels
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=(1 - train_ratio - val_ratio), random_state=seed
    )
    
    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
