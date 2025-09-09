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


def plot_architecture(save_path: Optional[str] = None, figsize: Tuple[int, int] = (20, 12),
                     show_binding_comparison: bool = True) -> None:
    """
    Create a comprehensive architectural diagram showing MultiOmicsBind with binding modality concept.
    
    Args:
        save_path (Optional[str]): Path to save the figure (default: None, shows plot)
        figsize (Tuple[int, int]): Figure size (default: (20, 12))
        show_binding_comparison (bool): Whether to show binding modality concept
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Professional color palette
    colors = {
        'input': '#F8F9FA',          # Light gray for inputs
        'input_border': '#343A40',   # Dark border
        'encoder': '#E3F2FD',        # Light blue for encoders  
        'encoder_border': '#1976D2', # Blue border
        'embedding': '#1976D2',      # Blue for embeddings
        'binding': '#FF6B35',        # Orange for binding modality
        'fusion': '#0D47A1',         # Dark blue for fusion
        'output': '#2E7D32',         # Green for outputs
        'loss': '#D32F2F',           # Red for losses
        'text_dark': '#212121',      # Dark text
        'text_light': '#FFFFFF',     # Light text
        'arrow': '#424242',          # Dark gray arrows
        'contrastive': '#FF5722',    # Orange for contrastive connections
        'binding_arrow': '#FF6B35'   # Orange for binding arrows
    }
    
    # Font settings
    title_font = {'fontsize': 22, 'fontweight': 'bold', 'fontfamily': 'Arial'}
    subtitle_font = {'fontsize': 16, 'fontweight': 'normal', 'fontfamily': 'Arial', 'style': 'italic'}
    label_font = {'fontsize': 11, 'fontweight': 'bold', 'fontfamily': 'Arial'}
    small_font = {'fontsize': 9, 'fontweight': 'normal', 'fontfamily': 'Arial'}
    
    # Main title
    ax.text(10, 11.5, 'MultiOmicsBind Architecture', ha='center', va='center', 
           color=colors['text_dark'], **title_font)
    ax.text(10, 11, 'Binding Modality Approach for Efficient Multi-Omics Integration', 
           ha='center', va='center', color=colors['text_dark'], **subtitle_font)
    
    # Define modalities
    modalities = [
        ('Transcriptomics\n(20K genes)', 'transcriptomics'),
        ('Proteomics\n(8K proteins)', 'proteomics'), 
        ('Metabolomics\n(2.5K metabolites)', 'metabolomics'),
        ('Cell Painting\n(1.5K features)', 'cell_painting'),
        ('Genomics\n(500K SNPs)', 'genomics')
    ]
    
    # Layer positions
    input_y = 9.5
    encoder_y = 8.2
    embedding_y = 6.8
    binding_y = 5.2
    fusion_y = 3.8
    output_y = 2.0
    
    # Box dimensions
    box_width, box_height = 2.2, 0.7
    
    # Draw input data layer
    ax.text(1, input_y + 0.5, 'Input Data', ha='left', va='center', 
           color=colors['text_dark'], **label_font)
    
    x_positions = np.linspace(3, 17, len(modalities))
    
    for i, ((mod_text, mod_name), x) in enumerate(zip(modalities, x_positions)):
        # Input box
        rect = patches.FancyBboxPatch(
            (x - box_width/2, input_y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.1", facecolor=colors['input'],
            edgecolor=colors['input_border'], linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, input_y, mod_text, ha='center', va='center', 
               color=colors['text_dark'], **small_font)
        
        # Encoder
        rect = patches.FancyBboxPatch(
            (x - box_width/2, encoder_y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.1", facecolor=colors['encoder'],
            edgecolor=colors['encoder_border'], linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, encoder_y, f'{mod_name.title()}\nEncoder', ha='center', va='center', 
               color=colors['text_dark'], **small_font)
        
        # Embedding
        embed_color = colors['binding'] if i == 0 else colors['embedding']  # First is binding modality
        rect = patches.FancyBboxPatch(
            (x - box_width/2, embedding_y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.1", facecolor=embed_color,
            edgecolor='white', linewidth=2
        )
        ax.add_patch(rect)
        
        embed_text = 'Binding Embedding\n(768-dim)' if i == 0 else 'Embedding\n(768-dim)'
        text_color = colors['text_light']
        ax.text(x, embedding_y, embed_text, ha='center', va='center', 
               color=text_color, **small_font)
        
        # Arrows: input -> encoder -> embedding
        ax.arrow(x, input_y - box_height/2, 0, -0.4, head_width=0.15, head_length=0.1,
                fc=colors['arrow'], ec=colors['arrow'], linewidth=2)
        ax.arrow(x, encoder_y - box_height/2, 0, -0.4, head_width=0.15, head_length=0.1,
                fc=colors['arrow'], ec=colors['arrow'], linewidth=2)
    
    # Binding modality concept
    ax.text(1, binding_y + 0.5, 'Binding Strategy', ha='left', va='center', 
           color=colors['text_dark'], **label_font)
    
    # Binding modality arrows - all other modalities connect to the first (binding modality)
    binding_x = x_positions[0]
    
    for i, x in enumerate(x_positions[1:], 1):
        # Curved arrow from each modality to binding modality
        ax.annotate('', xy=(binding_x, binding_y), xytext=(x, embedding_y - box_height/2),
                   arrowprops=dict(arrowstyle='->', color=colors['binding_arrow'], 
                                 lw=3, connectionstyle="arc3,rad=0.3"))
    
    # Add binding modality label
    ax.text(binding_x, binding_y, 'Transcriptomics\n(Binding Modality)', 
           ha='center', va='center', color=colors['binding'], 
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    edgecolor=colors['binding'], linewidth=2))
    
    # Multi-modal fusion
    fusion_x = 10
    rect = patches.FancyBboxPatch(
        (fusion_x - 3, fusion_y - box_height/2), 6, box_height,
        boxstyle="round,pad=0.1", facecolor=colors['fusion'],
        edgecolor='white', linewidth=2
    )
    ax.add_patch(rect)
    ax.text(fusion_x, fusion_y, 'Multi-Modal Fusion Layer\n(Combines all embeddings)', 
           ha='center', va='center', color=colors['text_light'], **label_font)
    
    # Arrows from embeddings to fusion
    for x in x_positions:
        ax.annotate('', xy=(fusion_x, fusion_y + box_height/2), 
                   xytext=(x, embedding_y - box_height/2),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                                 lw=2, connectionstyle="arc3,rad=0.1"))
    
    # Output layer
    outputs = [
        ('Contrastive\nLearning', fusion_x - 2.5),
        ('Classification\nHead (Optional)', fusion_x + 2.5)
    ]
    
    for text, x in outputs:
        rect = patches.FancyBboxPatch(
            (x - box_width/2, output_y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.1", facecolor=colors['output'],
            edgecolor='white', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, output_y, text, ha='center', va='center', 
               color=colors['text_light'], **small_font)
        
        # Arrow from fusion to outputs
        ax.arrow(x, fusion_y - box_height/2, 0, -1.0, head_width=0.15, head_length=0.1,
                fc=colors['arrow'], ec=colors['arrow'], linewidth=2)
    
    # Add explanatory text boxes
    # Efficiency box - positioned to avoid overlap with Genomics encoder
    efficiency_text = (
        "Efficiency Comparison:\n"
        "• All-pairs: O(n²) complexity\n"
        "• Binding modality: O(n) complexity\n"
        "• 3-6x speedup with 5+ modalities"
    )
    ax.text(16.5, 10.5, efficiency_text, ha='center', va='top',
           fontsize=9, color=colors['text_dark'],
           bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F5E8', 
                    edgecolor=colors['output'], alpha=0.9))
    
    # Key insight box  
    insight_text = (
        "Key Insight:\n"
        "Use transcriptomics as anchor\n"
        "(most comprehensive readout)\n"
        "All other modalities align to it"
    )
    ax.text(18.5, 5.5, insight_text, ha='right', va='top',
           fontsize=9, color=colors['text_dark'],
           bbox=dict(boxstyle="round,pad=0.4", facecolor='#FFF3E0', 
                    edgecolor=colors['binding'], alpha=0.9))
    
    # Benefits box
    benefits_text = (
        "Benefits:\n"
        "• Faster training\n"
        "• More stable gradients\n"
        "• Better interpretability\n"
        "• Handles missing modalities"
    )
    ax.text(18.5, 2.5, benefits_text, ha='right', va='top',
           fontsize=9, color=colors['text_dark'],
           bbox=dict(boxstyle="round,pad=0.4", facecolor='#E3F2FD', 
                    edgecolor=colors['encoder_border'], alpha=0.9))
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['binding'], label='Binding Modality (Anchor)'),
        patches.Patch(color=colors['embedding'], label='Other Modality Embeddings'),
        patches.Patch(color=colors['fusion'], label='Multi-Modal Fusion'),
        patches.Patch(color=colors['output'], label='Output Heads')
    ]
    
    legend = ax.legend(handles=legend_elements, 
                      loc='lower left', 
                      bbox_to_anchor=(0.02, 0.02), 
                      frameon=True, 
                      fancybox=True, 
                      shadow=True, 
                      fontsize=10,
                      title='Architecture Components',
                      title_fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Architecture diagram saved to {save_path}")
    else:
        plt.show()





def plot_training_history(
    history: Dict[str, List[float]], 
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot training history with loss curves and metrics.
    
    Args:
        history (Dict[str, List[float]]): Training history containing losses and metrics
        save_path (Optional[str]): Path to save the figure (default: None, shows plot)
        figsize (Tuple[int, int]): Figure size (default: (12, 8))
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    epochs = range(1, len(history['total_loss']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Contrastive loss
    if 'contrastive_loss' in history:
        axes[1].plot(epochs, history['contrastive_loss'], 'r-', linewidth=2, label='Contrastive Loss')
        axes[1].set_title('Contrastive Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    # Classification loss
    if 'classification_loss' in history:
        axes[2].plot(epochs, history['classification_loss'], 'g-', linewidth=2, label='Classification Loss')
        axes[2].set_title('Classification Loss', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    
    # Accuracy
    if 'accuracy' in history:
        axes[3].plot(epochs, history['accuracy'], 'purple', linewidth=2, label='Accuracy')
        axes[3].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Accuracy')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
    
    plt.suptitle('Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    else:
        plt.show()


def plot_embeddings_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot UMAP visualization of embeddings.
    
    Args:
        embeddings (np.ndarray): Embeddings to visualize
        labels (np.ndarray): Labels for coloring
        save_path (Optional[str]): Path to save the figure
        figsize (Tuple[int, int]): Figure size
    """
    try:
        import umap
        
        # Reduce dimensionality with UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=figsize)
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                            c=labels, cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(scatter)
        plt.title('UMAP Visualization of Embeddings', fontsize=16, fontweight='bold')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"UMAP plot saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("UMAP not installed. Please install with: pip install umap-learn")


def plot_feature_importance(
    importance_scores: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    top_k: int = 20,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot feature importance scores for different modalities.
    
    Args:
        importance_scores (Dict[str, np.ndarray]): Feature importance scores per modality
        save_path (Optional[str]): Path to save the figure
        top_k (int): Number of top features to show
        figsize (Tuple[int, int]): Figure size
    """
    n_modalities = len(importance_scores)
    fig, axes = plt.subplots(1, n_modalities, figsize=figsize)
    
    if n_modalities == 1:
        axes = [axes]
    
    for idx, (modality, scores) in enumerate(importance_scores.items()):
        # Get top k features
        top_indices = np.argsort(scores)[-top_k:]
        top_scores = scores[top_indices]
        
        # Create feature names
        feature_names = [f"{modality}_{i}" for i in top_indices]
        
        # Plot
        axes[idx].barh(range(len(top_scores)), top_scores)
        axes[idx].set_yticks(range(len(top_scores)))
        axes[idx].set_yticklabels(feature_names, fontsize=8)
        axes[idx].set_title(f'Top {top_k} Features - {modality}', fontweight='bold')
        axes[idx].set_xlabel('Importance Score')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
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
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
