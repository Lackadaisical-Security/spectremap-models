"""
Basic tests for Spectre Map models.
"""

import unittest
import numpy as np
import tensorflow as tf
from spectremap_models.models.cnn_model import SpectreMapCNN
from spectremap_models.models.rnn_model import SpectreMapRNN


class TestCNNModel(unittest.TestCase):
    """Test cases for CNN model."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def test_cnn_initialization(self):
        """Test CNN model initialization."""
        model = SpectreMapCNN(
            input_shape=(28, 28, 1),
            num_classes=10,
            name="test_cnn"
        )
        self.assertEqual(model.name, "test_cnn")
        self.assertEqual(model.input_shape, (28, 28, 1))
        self.assertEqual(model.num_classes, 10)
    
    def test_cnn_build(self):
        """Test CNN model building."""
        model = SpectreMapCNN(input_shape=(28, 28, 1), num_classes=10)
        keras_model = model.build_model(num_conv_blocks=2, filters_base=16)
        
        self.assertIsNotNone(keras_model)
        self.assertIsInstance(keras_model, tf.keras.Model)
        self.assertGreater(keras_model.count_params(), 0)
    
    def test_cnn_prediction(self):
        """Test CNN model prediction."""
        model = SpectreMapCNN(input_shape=(28, 28, 1), num_classes=10)
        model.build_model(num_conv_blocks=2)
        
        x_test = np.random.rand(5, 28, 28, 1).astype('float32')
        predictions = model.predict(x_test, verbose=0)
        
        self.assertEqual(predictions.shape, (5, 10))
        # Check predictions sum to 1 (softmax output)
        np.testing.assert_array_almost_equal(
            predictions.sum(axis=1),
            np.ones(5),
            decimal=5
        )


class TestRNNModel(unittest.TestCase):
    """Test cases for RNN model."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def test_rnn_initialization(self):
        """Test RNN model initialization."""
        model = SpectreMapRNN(
            input_shape=(50, 1),
            num_classes=None,
            name="test_rnn"
        )
        self.assertEqual(model.name, "test_rnn")
        self.assertEqual(model.input_shape, (50, 1))
        self.assertIsNone(model.num_classes)
        self.assertEqual(model.task_type, "regression")
    
    def test_rnn_build_lstm(self):
        """Test RNN model building with LSTM."""
        model = SpectreMapRNN(input_shape=(50, 1))
        keras_model = model.build_model(rnn_type='LSTM', num_layers=2, units=32)
        
        self.assertIsNotNone(keras_model)
        self.assertIsInstance(keras_model, tf.keras.Model)
        self.assertGreater(keras_model.count_params(), 0)
    
    def test_rnn_build_gru(self):
        """Test RNN model building with GRU."""
        model = SpectreMapRNN(input_shape=(50, 1))
        keras_model = model.build_model(rnn_type='GRU', num_layers=1, units=32)
        
        self.assertIsNotNone(keras_model)
        self.assertIsInstance(keras_model, tf.keras.Model)
    
    def test_rnn_prediction(self):
        """Test RNN model prediction."""
        model = SpectreMapRNN(input_shape=(50, 1))
        model.build_model(rnn_type='LSTM', num_layers=2, units=32)
        
        x_test = np.random.rand(5, 50, 1).astype('float32')
        predictions = model.predict(x_test, verbose=0)
        
        self.assertEqual(predictions.shape, (5, 1))
    
    def test_rnn_classification(self):
        """Test RNN model for classification."""
        model = SpectreMapRNN(input_shape=(50, 1), num_classes=5)
        model.build_model(rnn_type='LSTM', num_layers=1, units=32)
        
        x_test = np.random.rand(3, 50, 1).astype('float32')
        predictions = model.predict(x_test, verbose=0)
        
        self.assertEqual(predictions.shape, (3, 5))
        # Check predictions sum to 1 (softmax output)
        np.testing.assert_array_almost_equal(
            predictions.sum(axis=1),
            np.ones(3),
            decimal=5
        )


if __name__ == '__main__':
    unittest.main()
