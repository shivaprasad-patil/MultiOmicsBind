"""Training utilities and trainer classes."""

from .trainer import train_multiomicsbind, evaluate_model, EarlyStopping, train_temporal_model
from .evaluation import (
    evaluate_temporal_model, 
    compute_cross_modal_similarity,
    analyze_similarity_by_class
)
from .interpretation import get_gradients, compute_feature_importance

__all__ = [
    "train_multiomicsbind", 
    "evaluate_model", 
    "EarlyStopping",
    "train_temporal_model",
    "evaluate_temporal_model",
    "compute_cross_modal_similarity",
    "analyze_similarity_by_class",
    "get_gradients",
    "compute_feature_importance"
]
