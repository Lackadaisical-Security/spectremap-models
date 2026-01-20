"""
Recurrent Neural Network model for Spectre Map.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from spectremap_models.models.base_model import BaseModel
from typing import Tuple, Optional


class SpectreMapRNN(BaseModel):
    """
    Recurrent Neural Network for sequence-based tasks in Spectre Map system.
    
    This model can be used for time series prediction, sequence classification,
    or any sequential data processing.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (100, 1),
        num_classes: Optional[int] = None,
        output_steps: Optional[int] = None,
        name: str = "spectremap_rnn"
    ):
        """
        Initialize the RNN model.
        
        Args:
            input_shape: Shape of input sequences (timesteps, features)
            num_classes: Number of output classes (for classification)
            output_steps: Number of output timesteps (for sequence prediction)
            name: Name of the model
        """
        super().__init__(name=name)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.output_steps = output_steps
        
        # Determine task type
        if num_classes is not None:
            self.task_type = 'classification'
        elif output_steps is not None:
            self.task_type = 'sequence'
        else:
            self.task_type = 'regression'
    
    def build_model(
        self,
        rnn_type: str = 'LSTM',
        num_layers: int = 2,
        units: int = 64,
        dropout_rate: float = 0.2,
        recurrent_dropout: float = 0.2,
        dense_units: Optional[int] = None,
        learning_rate: float = 0.001,
        bidirectional: bool = False,
        **kwargs
    ) -> tf.keras.Model:
        """
        Build the RNN model architecture.
        
        Args:
            rnn_type: Type of RNN cell ('LSTM', 'GRU', or 'SimpleRNN')
            num_layers: Number of RNN layers
            units: Number of units in each RNN layer
            dropout_rate: Dropout rate
            recurrent_dropout: Recurrent dropout rate
            dense_units: Number of units in dense layer (optional)
            learning_rate: Learning rate for optimizer
            bidirectional: Whether to use bidirectional RNN
            **kwargs: Additional arguments
            
        Returns:
            Compiled Keras model
        """
        # Select RNN cell type
        rnn_cells = {
            'LSTM': layers.LSTM,
            'GRU': layers.GRU,
            'SimpleRNN': layers.SimpleRNN
        }
        
        if rnn_type not in rnn_cells:
            raise ValueError(f"RNN type {rnn_type} not supported. Choose from {list(rnn_cells.keys())}")
        
        RNNCell = rnn_cells[rnn_type]
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        
        # RNN layers
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1) or (self.task_type == 'sequence')
            
            rnn_layer = RNNCell(
                units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                name=f'{rnn_type.lower()}_{i}'
            )
            
            if bidirectional:
                x = layers.Bidirectional(rnn_layer, name=f'bidirectional_{i}')(x)
            else:
                x = rnn_layer(x)
        
        # Optional dense layer
        if dense_units is not None:
            x = layers.Dense(dense_units, activation='relu', name='dense')(x)
            x = layers.Dropout(dropout_rate, name='dropout')(x)
        
        # Output layer based on task type
        if self.task_type == 'classification':
            outputs = layers.Dense(
                self.num_classes,
                activation='softmax' if self.num_classes > 2 else 'sigmoid',
                name='output'
            )(x)
            loss = 'sparse_categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
            metrics = ['accuracy']
            
        elif self.task_type == 'sequence':
            outputs = layers.TimeDistributed(
                layers.Dense(1, activation='linear'),
                name='output'
            )(x)
            loss = 'mse'
            metrics = ['mae']
            
        else:  # regression
            outputs = layers.Dense(1, activation='linear', name='output')(x)
            loss = 'mse'
            metrics = ['mae']
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        return model
    
    def build_attention_model(
        self,
        rnn_type: str = 'LSTM',
        units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        **kwargs
    ) -> tf.keras.Model:
        """
        Build an RNN model with attention mechanism.
        
        Args:
            rnn_type: Type of RNN cell ('LSTM' or 'GRU')
            units: Number of units in RNN layer
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
            **kwargs: Additional arguments
            
        Returns:
            Compiled Keras model with attention
        """
        # Select RNN cell type
        rnn_cells = {
            'LSTM': layers.LSTM,
            'GRU': layers.GRU
        }
        
        if rnn_type not in rnn_cells:
            raise ValueError(f"RNN type {rnn_type} not supported for attention. Choose from {list(rnn_cells.keys())}")
        
        RNNCell = rnn_cells[rnn_type]
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        
        # RNN layer with return sequences
        rnn_out = RNNCell(
            units,
            return_sequences=True,
            dropout=dropout_rate,
            name=f'{rnn_type.lower()}'
        )(inputs)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh', name='attention_score')(rnn_out)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax', name='attention_weights')(attention)
        attention = layers.RepeatVector(units)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        context = layers.Multiply(name='attention_mul')([rnn_out, attention])
        context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='attention_sum')(context)
        
        # Output layer
        x = layers.Dropout(dropout_rate)(context)
        
        if self.task_type == 'classification':
            outputs = layers.Dense(
                self.num_classes,
                activation='softmax' if self.num_classes > 2 else 'sigmoid',
                name='output'
            )(x)
            loss = 'sparse_categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            outputs = layers.Dense(1, activation='linear', name='output')(x)
            loss = 'mse'
            metrics = ['mae']
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name=f"{self.name}_attention")
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        return model
