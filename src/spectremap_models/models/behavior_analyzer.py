"""
Behavioral Analysis Model for Spectre Map.

This model uses LSTM to profile entity behavior and detect deviations
from normal patterns.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from spectremap_models.models.base_model import BaseModel
from typing import Tuple


class BehaviorAnalyzer(BaseModel):
    """
    LSTM-based behavioral analysis for entity profiling.
    
    Designed for Spectre Map to track and analyze:
    - User behavior patterns
    - Device activity sequences
    - Lateral movement detection
    - Insider threat identification
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (50, 20),
        num_behavior_classes: int = 5,
        name: str = "behavior_analyzer"
    ):
        """
        Initialize the behavior analyzer.
        
        Args:
            input_shape: Shape of behavior sequences (timesteps, features)
            num_behavior_classes: Number of behavior categories
            name: Name of the model
        """
        super().__init__(name=name)
        self.input_shape = input_shape
        self.num_behavior_classes = num_behavior_classes
        
    def build_model(
        self,
        lstm_units: list = [128, 64],
        dense_units: int = 64,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.0001,
        **kwargs
    ) -> tf.keras.Model:
        """
        Build the behavioral analysis model.
        
        Args:
            lstm_units: List of LSTM units for each layer
            dense_units: Number of units in dense layer
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        
        # Bidirectional LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)
            x = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate * 0.5,
                    name=f'lstm_{i}'
                ),
                name=f'bidirectional_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
        
        # Dense layers
        x = layers.Dense(dense_units, activation='relu', name='dense')(x)
        x = layers.Dropout(dropout_rate, name='dropout')(x)
        
        # Multi-class output for behavior classification
        outputs = layers.Dense(
            self.num_behavior_classes,
            activation='softmax',
            name='output'
        )(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def export_for_spectremap(self, export_path: str):
        """
        Export model in TensorFlow SavedModel format for Spectre Map.
        
        Args:
            export_path: Path to save the model (e.g., 'behavior_lstm')
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Export using Keras save (creates SavedModel format by default)
        self.model.save(export_path)
        print(f"Model exported for Spectre Map at: {export_path}")
