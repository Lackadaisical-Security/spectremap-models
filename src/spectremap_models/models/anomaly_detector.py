"""
Anomaly Detection Model for Spectre Map.

This model detects network traffic anomalies and suspicious patterns
using a Convolutional Neural Network architecture.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from spectremap_models.models.base_model import BaseModel
from typing import Tuple


class AnomalyDetector(BaseModel):
    """
    CNN-based anomaly detection for network traffic analysis.
    
    Designed for Spectre Map's NetSpectre module to identify:
    - Port scanning activities
    - DDoS attack patterns
    - Unusual traffic flows
    - Protocol anomalies
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (100, 10),
        name: str = "anomaly_detector"
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            input_shape: Shape of input traffic features (timesteps, features)
            name: Name of the model
        """
        super().__init__(name=name)
        self.input_shape = input_shape
        
    def build_model(
        self,
        conv_filters: list = [64, 128, 256],
        dense_units: int = 128,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.0001,
        **kwargs
    ) -> tf.keras.Model:
        """
        Build the anomaly detection model.
        
        Args:
            conv_filters: List of filter sizes for conv layers
            dense_units: Number of units in dense layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # Reshape for 1D convolution
        x = layers.Reshape((self.input_shape[0], self.input_shape[1], 1))(inputs)
        
        # Convolutional layers for feature extraction
        for i, filters in enumerate(conv_filters):
            x = layers.Conv2D(
                filters,
                (3, 3),
                padding='same',
                activation='relu',
                name=f'conv_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
            x = layers.MaxPooling2D((2, 2), name=f'pool_{i}')(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        
        # Dense layers
        x = layers.Dense(dense_units, activation='relu', name='dense')(x)
        x = layers.Dropout(dropout_rate, name='dropout_final')(x)
        
        # Output: Binary classification (normal vs anomaly)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def export_for_spectremap(self, export_path: str):
        """
        Export model in TensorFlow SavedModel format for Spectre Map.
        
        Args:
            export_path: Path to save the model (e.g., 'anomaly_detector')
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.save(export_path, save_format='tf')
        print(f"Model exported for Spectre Map at: {export_path}")
