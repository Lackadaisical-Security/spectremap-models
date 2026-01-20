"""
Signal Classification Model for Spectre Map.

This model classifies RF signals and wireless protocols for the
SignalScope module.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from spectremap_models.models.base_model import BaseModel
from typing import Tuple


class SignalClassifier(BaseModel):
    """
    CNN-based signal classification for RF analysis.
    
    Designed for Spectre Map's SignalScope module to identify:
    - WiFi protocols and standards
    - Bluetooth device types
    - Zigbee networks
    - SDR signal patterns
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (128, 128),
        num_signal_types: int = 10,
        name: str = "signal_classifier"
    ):
        """
        Initialize the signal classifier.
        
        Args:
            input_shape: Shape of signal spectrograms (height, width)
            num_signal_types: Number of signal types to classify
            name: Name of the model
        """
        super().__init__(name=name)
        self.input_shape = input_shape + (1,)  # Add channel dimension
        self.num_signal_types = num_signal_types
        
    def build_model(
        self,
        conv_blocks: int = 4,
        filters_base: int = 32,
        dense_units: int = 256,
        dropout_rate: float = 0.4,
        learning_rate: float = 0.0001,
        **kwargs
    ) -> tf.keras.Model:
        """
        Build the signal classification model.
        
        Args:
            conv_blocks: Number of convolutional blocks
            filters_base: Base number of filters
            dense_units: Number of units in dense layer
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        
        # Convolutional blocks for spectrogram analysis
        for i in range(conv_blocks):
            filters = filters_base * (2 ** i)
            
            x = layers.Conv2D(
                filters,
                (3, 3),
                padding='same',
                activation='relu',
                name=f'conv_{i}_1'
            )(x)
            x = layers.Conv2D(
                filters,
                (3, 3),
                padding='same',
                activation='relu',
                name=f'conv_{i}_2'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
            x = layers.MaxPooling2D((2, 2), name=f'pool_{i}')(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        
        # Dense layers
        x = layers.Dense(dense_units, activation='relu', name='dense_1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense_1')(x)
        x = layers.Dense(dense_units // 2, activation='relu', name='dense_2')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense_2')(x)
        
        # Multi-class output
        outputs = layers.Dense(
            self.num_signal_types,
            activation='softmax',
            name='output'
        )(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def export_for_spectremap(self, export_path: str):
        """
        Export model in TensorFlow SavedModel format for Spectre Map.
        
        Args:
            export_path: Path to save the model (e.g., 'signal_classifier')
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.save(export_path, save_format='tf')
        print(f"Model exported for Spectre Map at: {export_path}")
