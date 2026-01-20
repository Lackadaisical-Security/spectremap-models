"""
Visualization utilities for Spectre Map models.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List


def plot_training_history(
    history,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None
):
    """
    Plot training history.
    
    Args:
        history: Keras History object or dictionary
        metrics: List of metrics to plot (default: all metrics)
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    if hasattr(history, 'history'):
        history = history.history
    
    if metrics is None:
        # Get all metrics except validation metrics
        metrics = [k for k in history.keys() if not k.startswith('val_')]
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.plot(history[metric], label=f'Train {metric}')
        
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Val {metric}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: Optional[List[str]] = None,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names or range(len(cm)),
        yticklabels=class_names or range(len(cm)),
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_predictions(
    y_true,
    y_pred,
    num_samples: int = 100,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot predictions vs true values for regression tasks.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        num_samples: Number of samples to plot
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Predictions vs True values
    indices = range(min(num_samples, len(y_true)))
    ax1.plot(indices, y_true[:num_samples], label='True', marker='o', alpha=0.7)
    ax1.plot(indices, y_pred[:num_samples], label='Predicted', marker='x', alpha=0.7)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Value')
    ax1.set_title('Predictions vs True Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.5)
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predictions')
    ax2.set_title('Prediction Scatter Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_model_architecture(model, save_path: str = "model_architecture.png"):
    """
    Visualize model architecture.
    
    Args:
        model: Keras model
        save_path: Path to save the visualization
    """
    from tensorflow.keras.utils import plot_model
    
    plot_model(
        model,
        to_file=save_path,
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96
    )
    
    print(f"Model architecture saved to {save_path}")
