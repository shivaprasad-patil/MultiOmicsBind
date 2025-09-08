"""
MultiOmicsBind: A Deep Learning Framework for Multi-Omics Data Integration

This package provides a PyTorch-based framework for integrating and analyzing
multi-omics data using contrastive learning and neural encoders.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .core.model import MultiOmicsBindWithHead
from .core.encoders import OmicsEncoder, MetadataEncoder
from .core.losses import contrastive_loss
from .data.dataset import MultiOmicsDataset
from .training.trainer import train_multiomicsbind, evaluate_model
from .utils.visualization import plot_architecture, plot_training_history

__all__ = [
    "MultiOmicsBindWithHead",
    "OmicsEncoder", 
    "MetadataEncoder",
    "contrastive_loss",
    "MultiOmicsDataset",
    "train_multiomicsbind",
    "evaluate_model",
    "plot_architecture",
    "plot_training_history"
]
