"""
MultiOmicsBind: A Deep Learning Framework for Multi-Omics Data Integration

This package provides a PyTorch-based framework for integrating and analyzing
multi-omics data using contrastive learning and neural encoders.
"""

__version__ = "0.1.2"
__author__ = "Shivaprasad Patil"

from .core.model import MultiOmicsBindWithHead, TemporalMultiOmicsBind
from .core.encoders import OmicsEncoder, MetadataEncoder
from .core.temporal_encoders import (
    LSTMTemporalEncoder,             # Recommended for biological time series
    TransformerTemporalEncoder,      # For long sequences or complex patterns  
    AttentionPoolingTemporalEncoder, # For interpretability
    TemporalAggregationEncoder,      # For simple temporal aggregation
    create_temporal_encoder          # Factory function
)
from .core.losses import contrastive_loss, binding_modality_loss
from .data.dataset import MultiOmicsDataset, TemporalMultiOmicsDataset
from .training.trainer import train_multiomicsbind, evaluate_model
from .utils.visualization import (
    plot_architecture, 
    plot_training_history, 
    plot_embeddings_umap, 
    plot_feature_importance, 
    plot_confusion_matrix,
    plot_temporal_patterns,
    plot_temporal_heatmap,
    plot_temporal_architecture
)

__all__ = [
    # Core models
    "MultiOmicsBindWithHead",
    "TemporalMultiOmicsBind",
    
    # Encoders
    "OmicsEncoder", 
    "MetadataEncoder",
    "LSTMTemporalEncoder",
    "TransformerTemporalEncoder", 
    "AttentionPoolingTemporalEncoder",
    "TemporalAggregationEncoder",
    "create_temporal_encoder",
    
    # Loss functions
    "contrastive_loss",
    "binding_modality_loss",
    
    # Datasets
    "MultiOmicsDataset",
    "TemporalMultiOmicsDataset",
    
    # Training utilities
    "train_multiomicsbind",
    "evaluate_model",
    
    # Visualization
    "plot_architecture",
    "plot_training_history",
    "plot_embeddings_umap",
    "plot_feature_importance",
    "plot_confusion_matrix",
    "plot_temporal_patterns",
    "plot_temporal_heatmap",
    "plot_temporal_architecture"
]