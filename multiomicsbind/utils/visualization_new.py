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


def plot_architecture(save_path: Optional[str] = None, figsize: Tuple[int, int] = (18, 14),
                     custom_modalities: Optional[Dict[str, int]] = None) -> None:
    """
    Create a professional architectural diagram of the MultiOmicsBind model.
    
    Args:
        save_path (Optional[str]): Path to save the figure (default: None, shows plot)
        figsize (Tuple[int, int]): Figure size (default: (18, 14))
        custom_modalities (Optional[Dict[str, int]]): Custom modalities and their feature counts
                                                     (default: None, uses example modalities)
    """
    # Create figure with high DPI for crisp output
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Professional color palette with excellent contrast
    colors = {
        'input': '#F8F9FA',          # Light gray for inputs
        'input_border': '#343A40',   # Dark border
        'encoder': '#E3F2FD',        # Light blue for encoders  
        'encoder_border': '#1976D2', # Blue border
        'embedding': '#1976D2',      # Blue for embeddings
        'fusion': '#0D47A1',         # Dark blue for fusion
        'output': '#2E7D32',         # Green for outputs
        'loss': '#D32F2F',           # Red for losses
        'text_dark': '#212121',      # Dark text
        'text_light': '#FFFFFF',     # Light text
        'arrow': '#424242',          # Dark gray arrows
        'contrastive': '#FF5722'     # Orange for contrastive connections
    }
    
    # Standard font settings
    title_font = {'fontsize': 20, 'fontweight': 'bold', 'fontfamily': 'Arial'}
    label_font = {'fontsize': 11, 'fontweight': 'bold', 'fontfamily': 'Arial'}
    small_font = {'fontsize': 9, 'fontweight': 'normal', 'fontfamily': 'Arial'}
    
    # Use custom modalities or default examples
    if custom_modalities is None:
        modalities = {
            'Transcriptomics': '20K genes',
            'Proteomics': '8K proteins', 
            'Metabolomics': '2.5K metabolites',
            'Cell Painting': '1.5K features',
            'Genomics': '500K SNPs'
        }
    else:
        modalities = {name: f'{count:,} features' for name, count in custom_modalities.items()}
    
    # Title with perfect positioning
    ax.text(8, 13.2, 'MultiOmicsBind Architecture', ha='center', va='center', **title_font)
    ax.text(8, 12.7, '(Supports any number of modalities with any number of features)', 
           ha='center', va='center', fontsize=14, style='italic', color=colors['text_dark'])
    
    # Grid-based positioning for perfect alignment
    n_modalities = len(modalities)
    start_x = 2
    end_x = 14
    spacing = (end_x - start_x) / (n_modalities + 1)
    
    # Layer Y positions (grid-aligned)
    input_y = 11.5
    encoder_y = 10
    embedding_y = 8.5
    fusion_y = 6.5
    output_y = 4.8
    loss_y = 3.2
    
    # Standard box dimensions
    input_width, input_height = 1.8, 0.8
    encoder_width, encoder_height = 1.4, 0.6
    embed_width, embed_height = 1.4, 0.6
    fusion_width, fusion_height = 4, 0.8
    output_width, output_height = 2.2, 0.8
    loss_width, loss_height = 1.8, 0.6
    
    # Input data boxes with perfect grid alignment
    inputs = []
    for i, (name, desc) in enumerate(modalities.items()):
        x = start_x + (i + 1) * spacing
        inputs.append((f'{name}\\n({desc})', x, input_y))
    
    # Add metadata box (positioned to the right)
    metadata_x = 15
    inputs.append(('Metadata\\n(experimental data)', metadata_x, input_y))
    
    # Draw input boxes with consistent styling
    for text, x, y in inputs:
        rect = patches.FancyBboxPatch(
            (x - input_width/2, y - input_height/2), input_width, input_height,
            boxstyle="round,pad=0.1",
            facecolor=colors['input'],
            edgecolor=colors['input_border'],
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', color=colors['text_dark'], **label_font)
    
    # Encoder boxes with grid alignment
    encoders = []
    for i, _ in enumerate(modalities.items()):
        x = start_x + (i + 1) * spacing
        encoders.append(('Omics\\nEncoder', x, encoder_y))
    encoders.append(('Metadata\\nEncoder', metadata_x, encoder_y))
    
    for text, x, y in encoders:
        rect = patches.FancyBboxPatch(
            (x - encoder_width/2, y - encoder_height/2), encoder_width, encoder_height,
            boxstyle="round,pad=0.08",
            facecolor=colors['encoder'],
            edgecolor=colors['encoder_border'],
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', color=colors['text_dark'], **small_font)
    
    # Embedding boxes with consistent alignment
    embeddings = []
    for i, _ in enumerate(modalities.items()):
        x = start_x + (i + 1) * spacing
        embeddings.append(('Embedding\\n(768-dim)', x, embedding_y))
    embeddings.append(('Metadata\\nEmbedding', metadata_x, embedding_y))
    
    for text, x, y in embeddings:
        rect = patches.FancyBboxPatch(
            (x - embed_width/2, y - embed_height/2), embed_width, embed_height,
            boxstyle="round,pad=0.08",
            facecolor=colors['embedding'],
            edgecolor='white',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', color=colors['text_light'], **small_font)
    
    # Contrastive learning connections (elegant dotted lines)
    for i in range(len(embeddings)-1):  # Exclude metadata embedding
        for j in range(i+1, len(embeddings)-1):
            x1, y1 = embeddings[i][1], embeddings[i][2]
            x2, y2 = embeddings[j][1], embeddings[j][2]
            ax.plot([x1, x2], [y1, y2], color=colors['contrastive'], linestyle='--', 
                   alpha=0.8, linewidth=2.5)
    
    # Fusion layer (centered and prominent)
    fusion_x = 8  # Center of diagram
    fusion_rect = patches.FancyBboxPatch(
        (fusion_x - fusion_width/2, fusion_y - fusion_height/2), fusion_width, fusion_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['fusion'],
        edgecolor='white',
        linewidth=2
    )
    ax.add_patch(fusion_rect)
    ax.text(fusion_x, fusion_y, f'Multi-Modal Fusion (Mean Pooling)\\nCombines {len(modalities)} modalities + metadata', 
           ha='center', va='center', color=colors['text_light'], **label_font)
    
    # Output branches (symmetrically positioned)
    outputs = [
        ('Contrastive\\nLearning', fusion_x - 3, output_y),
        ('Classification\\nHead (Optional)', fusion_x + 3, output_y)
    ]
    
    for text, x, y in outputs:
        rect = patches.FancyBboxPatch(
            (x - output_width/2, y - output_height/2), output_width, output_height,
            boxstyle="round,pad=0.1",
            facecolor=colors['output'],
            edgecolor='white',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', color=colors['text_light'], **label_font)
    
    # Loss functions (aligned with outputs)
    losses = [
        ('InfoNCE Loss', fusion_x - 3, loss_y),
        ('Cross-Entropy Loss\\n(Optional)', fusion_x + 3, loss_y)
    ]
    
    for text, x, y in losses:
        rect = patches.FancyBboxPatch(
            (x - loss_width/2, y - loss_height/2), loss_width, loss_height,
            boxstyle="round,pad=0.08",
            facecolor=colors['loss'],
            edgecolor='white',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', color=colors['text_light'], **small_font)
    
    # PERFECT ARROW SYSTEM with consistent styling
    arrow_props = {
        'head_width': 0.15,
        'head_length': 0.12,
        'fc': colors['arrow'],
        'ec': colors['arrow'],
        'linewidth': 2.5,
        'alpha': 0.9
    }
    
    # 1. Input to encoders (perfectly vertical arrows)
    for _, x, y in inputs:
        target_y = encoder_y + encoder_height/2
        ax.arrow(x, y - input_height/2, 0, target_y - (y - input_height/2) - 0.1, **arrow_props)
    
    # 2. Encoders to embeddings (perfectly vertical arrows)
    for _, x, y in encoders:
        target_y = embedding_y + embed_height/2
        ax.arrow(x, y - encoder_height/2, 0, target_y - (y - encoder_height/2) - 0.1, **arrow_props)
    
    # 3. Embeddings to fusion (perfectly calculated converging arrows)
    fusion_center_x = fusion_x
    fusion_top_y = fusion_y + fusion_height/2
    
    for _, x, y in embeddings:
        start_y = y - embed_height/2
        # Calculate perfect arrow vector
        dx = fusion_center_x - x
        dy = fusion_top_y - start_y - 0.1
        
        # Adjust length to not overlap with fusion box
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx_adjusted = dx * 0.85  # Slightly shorter to avoid overlap
            dy_adjusted = dy * 0.85
            ax.arrow(x, start_y, dx_adjusted, dy_adjusted, **arrow_props)
    
    # 4. Fusion to outputs (perfectly symmetrical diverging arrows)
    fusion_bottom_y = fusion_y - fusion_height/2
    
    # Left arrow to contrastive learning
    left_target_x = fusion_x - 3
    left_target_y = output_y + output_height/2
    ax.arrow(fusion_x - 1, fusion_bottom_y, left_target_x - (fusion_x - 1), 
            left_target_y - fusion_bottom_y + 0.1, **arrow_props)
    
    # Right arrow to classification
    right_target_x = fusion_x + 3
    right_target_y = output_y + output_height/2
    ax.arrow(fusion_x + 1, fusion_bottom_y, right_target_x - (fusion_x + 1), 
            right_target_y - fusion_bottom_y + 0.1, **arrow_props)
    
    # 5. Outputs to losses (perfectly vertical arrows)
    for _, x, y in outputs:
        target_y = loss_y + loss_height/2
        ax.arrow(x, y - output_height/2, 0, target_y - (y - output_height/2) - 0.1, **arrow_props)
    
    # Professional annotations and labels
    
    # Contrastive learning connections annotation
    if len(embeddings) > 2:
        ax.text(1, embedding_y - 0.8, 'Contrastive\\nConnections', ha='center', va='center', 
                fontsize=10, weight='bold', color=colors['contrastive'],
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor=colors['contrastive'], alpha=0.9))
    
    # Scalability note (professional positioning)
    ax.text(15.5, 9.5, 'Scalable Architecture:\\n• Any number of modalities\\n• Any feature dimensions\\n• Flexible data types', 
           ha='right', va='center', fontsize=10, weight='bold', color=colors['text_dark'],
           bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F5E8', 
                    edgecolor='#4CAF50', alpha=0.9))
    
    # Professional legend with better positioning and styling
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input Data'),
        patches.Patch(color=colors['encoder'], label='Neural Encoders'),
        patches.Patch(color=colors['embedding'], label='Embeddings (768-dim)'), 
        patches.Patch(color=colors['fusion'], label='Multi-Modal Fusion'),
        patches.Patch(color=colors['output'], label='Output Heads'),
        patches.Patch(color=colors['loss'], label='Loss Functions'),
        patches.Patch(color=colors['contrastive'], label='Contrastive Learning'),
        patches.Patch(color=colors['arrow'], label='Data Flow')
    ]
    
    legend = ax.legend(handles=legend_elements, 
                      loc='upper left', 
                      bbox_to_anchor=(0.02, 0.98), 
                      frameon=True, 
                      fancybox=True, 
                      shadow=True, 
                      fontsize=10,
                      title='Components',
                      title_fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor(colors['input_border'])
    
    # Professional grid (subtle)
    ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5, color=colors['text_dark'])
    
    # Set aspect ratio and tight layout
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(pad=1.0)
    
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
