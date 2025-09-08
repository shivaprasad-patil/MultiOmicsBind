"""
Visualization utilities for MultiOmicsBind.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import pandas as pd


def plot_architecture(save_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Create an architectural diagram of the MultiOmicsBind model.
    
    Args:
        save_path (Optional[str]): Path to save the figure (default: None, shows plot)
        figsize (Tuple[int, int]): Figure size (default: (14, 10))
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'encoder': '#B3D9FF', 
        'embedding': '#4A90E2',
        'fusion': '#357ABD',
        'output': '#2E5F88',
        'loss': '#FF6B6B'
    }
    
    # Input data boxes
    inputs = [
        ('Transcriptomics\n(6000 genes)', 0.5, 8.5),
        ('Proteomics\n(4000 proteins)', 2.5, 8.5),
        ('Cell Painting\n(1500 features)', 4.5, 8.5),
        ('Metadata\n(drug, dose, cell line)', 6.5, 8.5)
    ]
    
    for text, x, y in inputs:
        rect = patches.FancyBboxPatch(
            (x-0.4, y-0.3), 0.8, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=colors['input'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold')
    
    # Encoder boxes
    encoders = [
        ('Omics\nEncoder', 1.5, 7),
        ('Omics\nEncoder', 3.5, 7),
        ('Omics\nEncoder', 5.5, 7),
        ('Metadata\nEncoder', 6.5, 7)
    ]
    
    for text, x, y in encoders:
        rect = patches.FancyBboxPatch(
            (x-0.3, y-0.2), 0.6, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=colors['encoder'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9)
    
    # Embedding space
    embeddings = [
        ('Embedding\n768-dim', 1.5, 5.5),
        ('Embedding\n768-dim', 3.5, 5.5), 
        ('Embedding\n768-dim', 5.5, 5.5),
        ('Embedding\n768-dim', 6.5, 5.5)
    ]
    
    for text, x, y in embeddings:
        rect = patches.FancyBboxPatch(
            (x-0.3, y-0.2), 0.6, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=colors['embedding'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, color='white', weight='bold')
    
    # Contrastive learning connections
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            x1, y1 = embeddings[i][1], embeddings[i][2]
            x2, y2 = embeddings[j][1], embeddings[j][2]
            ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.6, linewidth=1)
    
    # Fusion layer
    fusion_rect = patches.FancyBboxPatch(
        (3.5, 3.8), 2, 0.4,
        boxstyle="round,pad=0.05",
        facecolor=colors['fusion'],
        edgecolor='black',
        linewidth=1
    )
    ax.add_patch(fusion_rect)
    ax.text(4.5, 4, 'Multi-Modal Fusion\n(Mean Pooling)', ha='center', va='center', 
            fontsize=10, color='white', weight='bold')
    
    # Output branches
    outputs = [
        ('Contrastive\nLearning', 2, 2.5),
        ('Classification\nHead', 7, 2.5)
    ]
    
    for text, x, y in outputs:
        rect = patches.FancyBboxPatch(
            (x-0.5, y-0.3), 1, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=colors['output'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, color='white', weight='bold')
    
    # Loss functions
    losses = [
        ('InfoNCE Loss', 2, 1),
        ('Cross-Entropy Loss', 7, 1)
    ]
    
    for text, x, y in losses:
        rect = patches.FancyBboxPatch(
            (x-0.4, y-0.2), 0.8, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=colors['loss'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, color='white')
    
    # Arrows
    # Input to encoders
    for i, (_, x, y) in enumerate(inputs):
        if i < 3:  # Omics data
            ax.arrow(x, y-0.3, 0, -0.5, head_width=0.05, head_length=0.1, fc='black', ec='black')
        else:  # Metadata
            ax.arrow(x, y-0.3, 0, -1.2, head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    # Encoders to embeddings
    for _, x, y in encoders:
        ax.arrow(x, y-0.2, 0, -1.1, head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    # Embeddings to fusion
    for _, x, y in embeddings:
        ax.arrow(x, y-0.2, 4.5-x, 4.2-y, head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    # Fusion to outputs
    ax.arrow(4.5, 3.8, -2.2, -1, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(4.5, 3.8, 2.2, -1, head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    # Outputs to losses
    ax.arrow(2, 2.2, 0, -0.9, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(7, 2.2, 0, -0.9, head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    # Title and annotations
    ax.text(5, 9.5, 'MultiOmicsBind Architecture', ha='center', va='center', 
            fontsize=16, weight='bold')
    
    ax.text(0.5, 4.5, 'Contrastive\nLearning', ha='center', va='center', 
            fontsize=12, weight='bold', rotation=90, color='red')
    
    # Legend
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input Data'),
        patches.Patch(color=colors['encoder'], label='Encoders'),
        patches.Patch(color=colors['embedding'], label='Embeddings'), 
        patches.Patch(color=colors['fusion'], label='Fusion Layer'),
        patches.Patch(color=colors['output'], label='Output Heads'),
        patches.Patch(color=colors['loss'], label='Loss Functions')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture diagram saved to {save_path}")
    else:
        plt.show()


def plot_training_history(
    history: Dict[str, List[float]], 
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot training history curves.
    
    Args:
        history (Dict[str, List[float]]): Training history dictionary
        save_path (Optional[str]): Path to save the figure
        figsize (Tuple[int, int]): Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['total_loss']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['total_loss'], 'b-', linewidth=2)
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Contrastive loss
    axes[1].plot(epochs, history['contrastive_loss'], 'r-', linewidth=2)
    axes[1].set_title('Contrastive Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    
    # Classification loss
    axes[2].plot(epochs, history['classification_loss'], 'g-', linewidth=2)
    axes[2].set_title('Classification Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    else:
        plt.show()


def plot_embeddings_umap(
    embeddings: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot UMAP visualization of embeddings for each modality.
    
    Args:
        embeddings (Dict[str, np.ndarray]): Dictionary of embeddings for each modality
        labels (Optional[np.ndarray]): Sample labels for coloring
        save_path (Optional[str]): Path to save the figure
        figsize (Tuple[int, int]): Figure size
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return
    
    n_modalities = len(embeddings)
    fig, axes = plt.subplots(1, n_modalities, figsize=figsize)
    
    if n_modalities == 1:
        axes = [axes]
    
    for i, (modality, emb) in enumerate(embeddings.items()):
        # Fit UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(emb)
        
        # Plot
        if labels is not None:
            scatter = axes[i].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                    c=labels, cmap='tab10', alpha=0.7, s=20)
            plt.colorbar(scatter, ax=axes[i])
        else:
            axes[i].scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7, s=20)
        
        axes[i].set_title(f'{modality.title()} Embeddings')
        axes[i].set_xlabel('UMAP 1')
        axes[i].set_ylabel('UMAP 2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"UMAP embeddings saved to {save_path}")
    else:
        plt.show()


def plot_feature_importance(
    feature_weights: np.ndarray,
    feature_names: List[str],
    top_k: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot feature importance scores.
    
    Args:
        feature_weights (np.ndarray): Feature importance weights
        feature_names (List[str]): Names of features
        top_k (int): Number of top features to show
        title (str): Plot title
        save_path (Optional[str]): Path to save the figure
        figsize (Tuple[int, int]): Figure size
    """
    # Get top k features by absolute importance
    abs_weights = np.abs(feature_weights)
    top_indices = np.argsort(abs_weights)[-top_k:][::-1]
    
    top_weights = feature_weights[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['red' if w < 0 else 'blue' for w in top_weights]
    bars = ax.barh(range(len(top_weights)), top_weights, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(top_weights)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, weight) in enumerate(zip(bars, top_weights)):
        ax.text(weight + 0.01 * max(abs(top_weights)), i, f'{weight:.3f}', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (Optional[List[str]]): Names of classes
        save_path (Optional[str]): Path to save the figure
        figsize (Tuple[int, int]): Figure size
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    if class_names is not None:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
