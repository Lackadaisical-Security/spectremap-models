"""
Convolutional Neural Network model for Spectre Map.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from spectremap_models.models.base_model import BaseModel
from typing import Tuple, Optional


class SpectreMapCNN(BaseModel):
    """
    Convolutional Neural Network for image-based tasks in Spectre Map system.
    
    This model can be used for image classification, feature extraction,
    or as a component in more complex architectures.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 10,
        name: str = "spectremap_cnn"
    ):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
            name: Name of the model
        """
        super().__init__(name=name)
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def build_model(
        self,
        num_conv_blocks: int = 3,
        filters_base: int = 32,
        dense_units: int = 128,
        dropout_rate: float = 0.5,
        learning_rate: float = 0.001,
        **kwargs
    ) -> tf.keras.Model:
        """
        Build the CNN model architecture.
        
        Args:
            num_conv_blocks: Number of convolutional blocks
            filters_base: Base number of filters (doubles with each block)
            dense_units: Number of units in dense layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            **kwargs: Additional arguments
            
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        
        # Convolutional blocks
        for i in range(num_conv_blocks):
            filters = filters_base * (2 ** i)
            
            # First conv layer in block
            x = layers.Conv2D(
                filters,
                (3, 3),
                padding='same',
                activation='relu',
                name=f'conv_{i}_1'
            )(x)
            
            # Second conv layer in block
            x = layers.Conv2D(
                filters,
                (3, 3),
                padding='same',
                activation='relu',
                name=f'conv_{i}_2'
            )(x)
            
            # Max pooling
            x = layers.MaxPooling2D((2, 2), name=f'pool_{i}')(x)
            
            # Batch normalization
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
        
        # Flatten
        x = layers.Flatten(name='flatten')(x)
        
        # Dense layer
        x = layers.Dense(dense_units, activation='relu', name='dense')(x)
        x = layers.Dropout(dropout_rate, name='dropout')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax' if self.num_classes > 1 else 'sigmoid',
            name='output'
        )(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy' if self.num_classes > 1 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_transfer_learning_model(
        self,
        base_model_name: str = 'MobileNetV2',
        trainable_base: bool = False,
        dense_units: int = 128,
        dropout_rate: float = 0.5,
        learning_rate: float = 0.001
    ) -> tf.keras.Model:
        """
        Build a transfer learning model using a pre-trained base.
        
        Args:
            base_model_name: Name of the pre-trained model ('MobileNetV2', 'ResNet50', etc.)
            trainable_base: Whether to fine-tune the base model
            dense_units: Number of units in dense layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained base model
        base_models = {
            'MobileNetV2': keras.applications.MobileNetV2,
            'ResNet50': keras.applications.ResNet50,
            'VGG16': keras.applications.VGG16,
            'InceptionV3': keras.applications.InceptionV3,
        }
        
        if base_model_name not in base_models:
            raise ValueError(f"Base model {base_model_name} not supported. Choose from {list(base_models.keys())}")
        
        base_model = base_models[base_model_name](
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = trainable_base
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax' if self.num_classes > 1 else 'sigmoid'
        )(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=f"{self.name}_transfer")
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy' if self.num_classes > 1 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
