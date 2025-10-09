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
from .training.evaluation import (
    evaluate_temporal_model,
    compute_cross_modal_similarity,
    analyze_similarity_by_class
)
from .training.interpretation import get_gradients, compute_feature_importance
from .training import train_temporal_model
from .utils.visualization import (
    plot_architecture, 
    plot_training_history, 
    plot_embeddings_umap, 
    plot_feature_importance, 
    plot_confusion_matrix,
    plot_temporal_patterns,
    plot_temporal_heatmap,
    plot_temporal_architecture,
    plot_training_history_detailed,
    plot_cross_modal_similarity_matrices,
    plot_feature_importance_distribution
)
from .utils.helpers import fix_nan_values, check_nan_values, check_and_fix_all_nan_values
from .analysis import create_analysis_report

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
    "train_temporal_model",
    "evaluate_temporal_model",
    "compute_cross_modal_similarity",
    "analyze_similarity_by_class",
    "get_gradients",
    "compute_feature_importance",
    
    # Visualization
    "plot_architecture",
    "plot_training_history",
    "plot_embeddings_umap",
    "plot_feature_importance",
    "plot_confusion_matrix",
    "plot_temporal_patterns",
    "plot_temporal_heatmap",
    "plot_temporal_architecture",
    "plot_training_history_detailed",
    "plot_cross_modal_similarity_matrices",
    "plot_feature_importance_distribution",
    
    # Utilities
    "fix_nan_values",
    "check_nan_values",
    
    # Analysis workflows
    "create_analysis_report"
]