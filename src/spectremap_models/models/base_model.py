"""
Base model class for Spectre Map models.
"""

from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Optional, Dict, Any
import os


class BaseModel(ABC):
    """
    Abstract base class for all Spectre Map models.
    
    This class provides common functionality for model management,
    including saving, loading, training, and evaluation.
    """
    
    def __init__(self, name: str = "base_model"):
        """
        Initialize the base model.
        
        Args:
            name: Name of the model
        """
        self.name = name
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        
    @abstractmethod
    def build_model(self, **kwargs) -> tf.keras.Model:
        """
        Build the model architecture.
        
        Returns:
            Compiled TensorFlow Keras model
        """
        pass
    
    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[list] = None,
        **kwargs
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            **kwargs: Additional arguments for model.fit()
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None
        
        self.history = self.model.fit(
            x_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks or [],
            **kwargs
        )
        
        return self.history
    
    def evaluate(self, x_test, y_test, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            x_test: Test data
            y_test: Test labels
            **kwargs: Additional arguments for model.evaluate()
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        results = self.model.evaluate(x_test, y_test, **kwargs)
        
        # Convert results to dictionary
        metric_names = self.model.metrics_names
        return dict(zip(metric_names, results))
    
    def predict(self, x, **kwargs):
        """
        Make predictions.
        
        Args:
            x: Input data
            **kwargs: Additional arguments for model.predict()
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.model.predict(x, **kwargs)
    
    def save(self, filepath: str, save_format: str = "tf"):
        """
        Save the model.
        
        Args:
            filepath: Path to save the model
            save_format: Format to save ('tf' or 'h5')
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        self.model.save(filepath, save_format=save_format)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.model.summary()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            "name": self.name,
            "model_config": self.model.get_config() if self.model else None
        }
