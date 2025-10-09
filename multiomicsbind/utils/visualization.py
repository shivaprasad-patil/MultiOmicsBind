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
    figsize: Tuple[int, int] = (10, 8),
    class_names: Optional[List[str]] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot UMAP visualization of embeddings.
    
    Args:
        embeddings (np.ndarray): Embeddings to visualize
        labels (np.ndarray): Labels for coloring
        save_path (Optional[str]): Path to save the figure
        figsize (Tuple[int, int]): Figure size
        class_names (Optional[List[str]]): Names for each class/label
        title (Optional[str]): Custom title for the plot
    """
    try:
        import umap
        
        # Reduce dimensionality with UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Get unique labels and create discrete colors
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each class separately for proper legend
        for i, label in enumerate(unique_labels):
            mask = labels == label
            # Use class names if provided, otherwise default names
            if class_names and len(class_names) > label:
                label_name = class_names[label]
            else:
                label_name = f'Class {label}'
            
            plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                       c=[colors[i]], alpha=0.7, s=50, 
                       label=label_name)
        
        # Use custom title if provided, otherwise use default
        plot_title = title if title else 'UMAP Visualization of Embeddings'
        plt.title(plot_title, fontsize=16, fontweight='bold')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
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


def plot_temporal_patterns(
    temporal_data: np.ndarray,
    timepoints: List[float],
    labels: Optional[np.ndarray] = None,
    feature_indices: Optional[List[int]] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot temporal patterns for selected features across different classes.
    
    Args:
        temporal_data (np.ndarray): Temporal data of shape (samples, timepoints, features)
        timepoints (List[float]): Time points for x-axis
        labels (Optional[np.ndarray]): Class labels for grouping
        feature_indices (Optional[List[int]]): Which features to plot (default: first 6)
        class_names (Optional[List[str]]): Names for each class
        save_path (Optional[str]): Path to save the figure
        figsize (Tuple[int, int]): Figure size
    """
    if feature_indices is None:
        feature_indices = list(range(min(6, temporal_data.shape[2])))
    
    n_features = len(feature_indices)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.Set1(np.linspace(0, 1, 10))  # Up to 10 classes
    
    for idx, feature_idx in enumerate(feature_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        if labels is not None:
            # Plot by class
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                class_data = temporal_data[mask, :, feature_idx]
                
                # Calculate mean and std
                mean_pattern = np.nanmean(class_data, axis=0)
                std_pattern = np.nanstd(class_data, axis=0)
                
                # Plot mean with error bars
                label_name = class_names[label] if class_names and len(class_names) > label else f'Class {label}'
                ax.plot(timepoints, mean_pattern, color=colors[i], linewidth=2, 
                       marker='o', label=label_name)
                ax.fill_between(timepoints, 
                              mean_pattern - std_pattern, 
                              mean_pattern + std_pattern,
                              color=colors[i], alpha=0.2)
        else:
            # Plot all samples
            for sample_idx in range(min(20, temporal_data.shape[0])):  # Limit to 20 samples
                sample_data = temporal_data[sample_idx, :, feature_idx]
                ax.plot(timepoints, sample_data, alpha=0.3, linewidth=1)
        
        ax.set_title(f'Feature {feature_idx}', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Expression')
        ax.grid(True, alpha=0.3)
        
        if labels is not None and idx == 0:  # Add legend to first subplot
            ax.legend()
    
    # Hide empty subplots
    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Temporal Expression Patterns', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temporal patterns plot saved to {save_path}")
    else:
        plt.show()


def plot_temporal_heatmap(
    temporal_data: np.ndarray,
    timepoints: List[float],
    feature_names: Optional[List[str]] = None,
    sample_labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    top_k_features: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot heatmap of temporal data showing feature expression over time.
    
    Args:
        temporal_data (np.ndarray): Temporal data of shape (samples, timepoints, features)
        timepoints (List[float]): Time points
        feature_names (Optional[List[str]]): Names of features
        sample_labels (Optional[np.ndarray]): Sample class labels
        class_names (Optional[List[str]]): Names for each class
        top_k_features (int): Number of top varying features to show
        save_path (Optional[str]): Path to save the figure
        figsize (Tuple[int, int]): Figure size
    """
    # Calculate temporal variance for each feature
    temporal_variance = np.nanvar(temporal_data, axis=(0, 1))  # Variance across samples and time
    top_feature_indices = np.argsort(temporal_variance)[-top_k_features:]
    
    # Get average temporal patterns
    if sample_labels is not None:
        unique_labels = np.unique(sample_labels)
        n_classes = len(unique_labels)
        
        fig, axes = plt.subplots(1, n_classes, figsize=(figsize[0] * n_classes / 2, figsize[1]))
        if n_classes == 1:
            axes = [axes]
        
        for i, label in enumerate(unique_labels):
            mask = sample_labels == label
            class_data = temporal_data[mask, :, :]
            
            # Average across samples in this class
            mean_data = np.nanmean(class_data, axis=0)  # Shape: (timepoints, features)
            
            # Select top varying features
            heatmap_data = mean_data[:, top_feature_indices].T  # Features x Timepoints
            
            # Create heatmap
            im = axes[i].imshow(heatmap_data, aspect='auto', cmap='RdBu_r', 
                              interpolation='nearest')
            
            # Set labels
            class_name = class_names[label] if class_names and len(class_names) > label else f'Class {label}'
            axes[i].set_title(f'{class_name}', fontweight='bold')
            axes[i].set_xlabel('Time Points')
            axes[i].set_xticks(range(len(timepoints)))
            axes[i].set_xticklabels([f'{t}' for t in timepoints])
            
            if i == 0:
                axes[i].set_ylabel('Features')
                if feature_names:
                    selected_names = [feature_names[idx] for idx in top_feature_indices]
                    axes[i].set_yticks(range(0, len(selected_names), max(1, len(selected_names)//10)))
                    axes[i].set_yticklabels([selected_names[j] for j in range(0, len(selected_names), 
                                                                            max(1, len(selected_names)//10))], 
                                          fontsize=8)
            else:
                axes[i].set_yticks([])
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
    else:
        # Single heatmap for all samples
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Average across all samples
        mean_data = np.nanmean(temporal_data, axis=0)  # Shape: (timepoints, features)
        
        # Select top varying features
        heatmap_data = mean_data[:, top_feature_indices].T  # Features x Timepoints
        
        # Create heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        
        # Set labels
        ax.set_title('Temporal Expression Heatmap', fontweight='bold')
        ax.set_xlabel('Time Points')
        ax.set_ylabel('Features')
        ax.set_xticks(range(len(timepoints)))
        ax.set_xticklabels([f'{t}' for t in timepoints])
        
        if feature_names:
            selected_names = [feature_names[idx] for idx in top_feature_indices]
            ax.set_yticks(range(0, len(selected_names), max(1, len(selected_names)//10)))
            ax.set_yticklabels([selected_names[j] for j in range(0, len(selected_names), 
                                                               max(1, len(selected_names)//10))], 
                              fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temporal heatmap saved to {save_path}")
    else:
        plt.show()


def plot_temporal_architecture(
    static_modalities: List[str],
    temporal_modalities: List[str],
    temporal_encoders: Dict[str, str],
    binding_modality: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """
    Plot architecture diagram for temporal multi-omics model.
    
    Args:
        static_modalities (List[str]): Names of static modalities
        temporal_modalities (List[str]): Names of temporal modalities
        temporal_encoders (Dict[str, str]): Encoder types for temporal modalities
        binding_modality (str): Name of binding modality
        save_path (Optional[str]): Path to save the figure
        figsize (Tuple[int, int]): Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'static': '#E3F2FD',
        'static_border': '#1976D2',
        'temporal': '#FFF3E0',
        'temporal_border': '#FF6B35',
        'binding': '#FF6B35',
        'fusion': '#0D47A1',
        'output': '#2E7D32',
        'text_dark': '#212121',
        'text_light': '#FFFFFF'
    }
    
    # Title
    ax.text(8, 9.5, 'Temporal MultiOmicsBind Architecture', ha='center', va='center',
           fontsize=20, fontweight='bold', color=colors['text_dark'])
    ax.text(8, 9, 'Mixed Static and Temporal Multi-Omics Integration', ha='center', va='center',
           fontsize=14, style='italic', color=colors['text_dark'])
    
    # Calculate positions
    all_modalities = static_modalities + temporal_modalities
    n_total = len(all_modalities)
    x_positions = np.linspace(2, 14, n_total)
    
    # Input layer
    input_y = 7.5
    encoder_y = 6.0
    embedding_y = 4.5
    fusion_y = 2.5
    output_y = 1.0
    
    box_width, box_height = 1.8, 0.6
    
    # Draw modalities
    for i, (modality, x) in enumerate(zip(all_modalities, x_positions)):
        is_static = modality in static_modalities
        is_binding = modality == binding_modality
        
        # Input box
        color = colors['static'] if is_static else colors['temporal']
        border = colors['static_border'] if is_static else colors['temporal_border']
        
        rect = patches.FancyBboxPatch(
            (x - box_width/2, input_y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor=border, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, input_y, modality.replace('_', '\n'), ha='center', va='center',
               fontsize=9, fontweight='bold', color=colors['text_dark'])
        
        # Encoder box
        if is_static:
            encoder_text = 'Static\nEncoder'
            encoder_color = colors['static']
        else:
            encoder_type = temporal_encoders.get(modality, 'LSTM')
            encoder_text = f'{encoder_type.upper()}\nEncoder'
            encoder_color = colors['temporal']
        
        rect = patches.FancyBboxPatch(
            (x - box_width/2, encoder_y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.1", facecolor=encoder_color, edgecolor=border, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, encoder_y, encoder_text, ha='center', va='center',
               fontsize=8, fontweight='bold', color=colors['text_dark'])
        
        # Embedding box
        embed_color = colors['binding'] if is_binding else colors['static_border']
        embed_text = 'Binding\nEmbedding' if is_binding else 'Embedding'
        text_color = colors['text_light'] if is_binding else colors['text_light']
        
        rect = patches.FancyBboxPatch(
            (x - box_width/2, embedding_y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.1", facecolor=embed_color, edgecolor='white', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, embedding_y, embed_text, ha='center', va='center',
               fontsize=8, fontweight='bold', color=text_color)
        
        # Arrows
        ax.arrow(x, input_y - box_height/2, 0, -0.7, head_width=0.1, head_length=0.1,
                fc='gray', ec='gray', linewidth=1.5)
        ax.arrow(x, encoder_y - box_height/2, 0, -0.7, head_width=0.1, head_length=0.1,
                fc='gray', ec='gray', linewidth=1.5)
    
    # Fusion layer
    fusion_x = 8
    rect = patches.FancyBboxPatch(
        (fusion_x - 4, fusion_y - box_height/2), 8, box_height,
        boxstyle="round,pad=0.1", facecolor=colors['fusion'], edgecolor='white', linewidth=2
    )
    ax.add_patch(rect)
    ax.text(fusion_x, fusion_y, 'Multi-Modal Fusion Layer', ha='center', va='center',
           fontsize=12, fontweight='bold', color=colors['text_light'])
    
    # Arrows to fusion
    for x in x_positions:
        ax.annotate('', xy=(fusion_x, fusion_y + box_height/2),
                   xytext=(x, embedding_y - box_height/2),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                                 connectionstyle="arc3,rad=0.1"))
    
    # Output layer
    rect = patches.FancyBboxPatch(
        (fusion_x - 2, output_y - box_height/2), 4, box_height,
        boxstyle="round,pad=0.1", facecolor=colors['output'], edgecolor='white', linewidth=2
    )
    ax.add_patch(rect)
    ax.text(fusion_x, output_y, 'Classification Output', ha='center', va='center',
           fontsize=12, fontweight='bold', color=colors['text_light'])
    
    # Arrow to output
    ax.arrow(fusion_x, fusion_y - box_height/2, 0, -0.8, head_width=0.15, head_length=0.1,
            fc='gray', ec='gray', linewidth=2)
    
    # Add legends and explanations
    # Static vs Temporal legend
    legend_x = 1
    legend_y = 3.5
    
    # Static box
    rect = patches.FancyBboxPatch(
        (legend_x - 0.3, legend_y), 0.6, 0.3,
        boxstyle="round,pad=0.05", facecolor=colors['static'], 
        edgecolor=colors['static_border'], linewidth=1
    )
    ax.add_patch(rect)
    ax.text(legend_x + 0.8, legend_y + 0.15, 'Static Modalities\n(Single timepoint)', 
           fontsize=9, va='center')
    
    # Temporal box
    rect = patches.FancyBboxPatch(
        (legend_x - 0.3, legend_y - 0.7), 0.6, 0.3,
        boxstyle="round,pad=0.05", facecolor=colors['temporal'], 
        edgecolor=colors['temporal_border'], linewidth=1
    )
    ax.add_patch(rect)
    ax.text(legend_x + 0.8, legend_y - 0.55, 'Temporal Modalities\n(Multiple timepoints)', 
           fontsize=9, va='center')
    
    # Binding modality info
    binding_x = 15
    binding_y = 6
    ax.text(binding_x, binding_y, f'Binding Modality:\n{binding_modality}', 
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    edgecolor=colors['binding'], linewidth=2))
    
    # Temporal encoders info
    encoder_info = '\n'.join([f'{mod}: {enc}' for mod, enc in temporal_encoders.items()])
    ax.text(binding_x, binding_y - 1.5, f'Temporal Encoders:\n{encoder_info}', 
           ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    edgecolor=colors['temporal_border'], linewidth=2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Temporal architecture diagram saved to {save_path}")
    else:
        plt.show()


def plot_training_history_detailed(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot detailed training history with loss and accuracy curves.
    
    Creates a comprehensive visualization of the training process with both
    loss and accuracy curves for training and validation sets. Includes grid,
    proper scaling, and summary statistics.
    
    Args:
        history: Dictionary containing training history with keys:
            - 'train_loss': List of training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'train_acc': List of training accuracies per epoch
            - 'val_acc': List of validation accuracies per epoch
        save_path (Optional[str]): Path to save the figure. If None, displays the plot.
        figsize (Tuple[int, int]): Figure size (default: (15, 5))
    
    Example:
        >>> from multiomicsbind import train_temporal_model, plot_training_history_detailed
        >>> model, history = train_temporal_model(dataset, device, epochs=20)
        >>> plot_training_history_detailed(history, save_path='training_history.png')
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
    ax2.plot(epochs, history['val_acc'], label='Val Accuracy', marker='s', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Add summary text
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    final_val_acc = history['val_acc'][-1]
    
    summary_text = (f"Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})\n"
                   f"Final Val Acc: {final_val_acc:.4f}")
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()


def plot_cross_modal_similarity_matrices(
    similarity_matrices: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = 'viridis'
) -> None:
    """
    Visualize cross-modal similarity matrices as heatmaps.
    
    Creates heatmap visualizations of pairwise similarity between embeddings
    from different modalities. Useful for understanding how well different
    data types align in the learned embedding space.
    
    Args:
        similarity_matrices: Dictionary mapping comparison names (e.g., 'mod1_vs_mod2')
            to similarity matrices of shape (n_samples, n_samples)
        save_path (Optional[str]): Path to save the figure. If None, displays the plot.
        figsize (Optional[Tuple[int, int]]): Figure size. If None, auto-calculated based
            on number of comparisons.
        cmap (str): Colormap for heatmaps (default: 'viridis')
    
    Example:
        >>> from multiomicsbind import (evaluate_temporal_model, 
        ...                            compute_cross_modal_similarity,
        ...                            plot_cross_modal_similarity_matrices)
        >>> embeddings, labels, predictions = evaluate_temporal_model(model, dataset, device)
        >>> similarity_matrices = compute_cross_modal_similarity(embeddings)
        >>> plot_cross_modal_similarity_matrices(similarity_matrices, 
        ...                                      save_path='similarity_matrices.png')
    """
    n_comparisons = len(similarity_matrices)
    
    if figsize is None:
        figsize = (6 * n_comparisons, 5)
    
    fig, axes = plt.subplots(1, n_comparisons, figsize=figsize)
    if n_comparisons == 1:
        axes = [axes]
    
    for idx, (comparison, sim_matrix) in enumerate(similarity_matrices.items()):
        # Plot heatmap
        im = axes[idx].imshow(sim_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        axes[idx].set_title(f'Cross-Modal Similarity\n{comparison.replace("_", " ").title()}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Sample Index', fontsize=10)
        axes[idx].set_ylabel('Sample Index', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[idx], label='Cosine Similarity')
        
        # Add mean similarity annotation
        mean_sim = np.mean(sim_matrix)
        axes[idx].text(0.5, -0.15, f'Mean: {mean_sim:.4f}', 
                      transform=axes[idx].transAxes, ha='center', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cross-modal similarity matrices saved to {save_path}")
    else:
        plt.show()


def plot_feature_importance_distribution(
    importance_df: pd.DataFrame,
    top_k: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot feature importance distribution by modality with top features.
    
    Creates visualizations showing both the overall importance distribution
    across modalities and the top individual features.
    
    Args:
        importance_df: DataFrame from compute_feature_importance with columns:
            - 'modality': Name of the modality
            - 'feature_name': Name of the feature
            - 'importance': Importance score
        top_k (int): Number of top features to show (default: 20)
        save_path (Optional[str]): Path to save the figure. If None, displays the plot.
        figsize (Tuple[int, int]): Figure size (default: (12, 6))
    
    Example:
        >>> from multiomicsbind import (compute_feature_importance, 
        ...                            plot_feature_importance_distribution)
        >>> importance_dict, importance_df = compute_feature_importance(model, dataset, device)
        >>> plot_feature_importance_distribution(importance_df, top_k=30,
        ...                                     save_path='feature_importance.png')
    """
    # Get top features
    top_features = importance_df.nlargest(top_k, 'importance')
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Plot 1: Distribution by modality
    modality_stats = importance_df.groupby('modality')['importance'].agg(['mean', 'std', 'max'])
    modality_stats = modality_stats.sort_values('mean', ascending=False)
    
    x = np.arange(len(modality_stats))
    ax1.bar(x, modality_stats['mean'], yerr=modality_stats['std'], 
           capsize=5, color='steelblue', alpha=0.7, label='Mean ± Std')
    ax1.scatter(x, modality_stats['max'], color='red', s=100, zorder=5, label='Max')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(modality_stats.index, rotation=45, ha='right')
    ax1.set_xlabel('Modality', fontsize=11)
    ax1.set_ylabel('Feature Importance', fontsize=11)
    ax1.set_title('Importance Distribution by Modality', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Top features
    colors_map = plt.cm.Set3(np.linspace(0, 1, len(top_features['modality'].unique())))
    modality_to_color = {mod: colors_map[i] for i, mod in enumerate(top_features['modality'].unique())}
    colors = [modality_to_color[mod] for mod in top_features['modality']]
    
    y_pos = np.arange(len(top_features))
    ax2.barh(y_pos, top_features['importance'], color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{row['feature_name'][:30]} ({row['modality']})" 
                         for _, row in top_features.iterrows()], fontsize=8)
    ax2.set_xlabel('Importance Score', fontsize=11)
    ax2.set_title(f'Top {top_k} Most Important Features', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance distribution saved to {save_path}")
    else:
        plt.show()
