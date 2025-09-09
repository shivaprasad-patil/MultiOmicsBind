"""
Binding modality loss functions for MultiOmicsBind.

This module implements the binding modality approach inspired by ImageBind,
where one modality serves as an anchor for aligning all other modalities.
"""

import torch
import torch.nn.functional as F
from typing import Dict


def contrastive_loss(embeddings: Dict[str, torch.Tensor], 
                    binding_modality: str,
                    temperature: float = 0.07) -> torch.Tensor:
    """
    Compute binding modality contrastive loss.
    
    Uses one modality as an anchor and aligns all other modalities to it.
    This reduces computational complexity from O(n²) to O(n) and often
    leads to better performance and interpretability.
    
    Args:
        embeddings (Dict[str, torch.Tensor]): Dictionary mapping modality names to 
            their embeddings of shape (batch_size, embed_dim)
        binding_modality (str): Name of the modality to use as anchor
        temperature (float): Temperature parameter for softmax (default: 0.07)
        
    Returns:
        torch.Tensor: Scalar contrastive loss value
        
    Example:
        >>> embeddings = {
        ...     'transcriptomics': torch.randn(32, 768),
        ...     'proteomics': torch.randn(32, 768),
        ...     'metabolomics': torch.randn(32, 768)
        ... }
        >>> # Use transcriptomics as binding modality
        >>> loss = contrastive_loss(embeddings, 'transcriptomics', temperature=0.07)
    """
    if len(embeddings) < 2:
        return torch.tensor(0.0, device=next(iter(embeddings.values())).device)
    
    if binding_modality not in embeddings:
        raise ValueError(f"Binding modality '{binding_modality}' not found in embeddings. "
                        f"Available modalities: {list(embeddings.keys())}")
    
    device = next(iter(embeddings.values())).device
    
    # Get binding modality embeddings (anchor)
    anchor_embeddings = F.normalize(embeddings[binding_modality], p=2, dim=1)
    
    total_loss = 0.0
    num_modalities = 0
    
    # Align each other modality to the binding modality
    for modality, modality_embeddings in embeddings.items():
        if modality != binding_modality:
            # Normalize target modality embeddings
            target_embeddings = F.normalize(modality_embeddings, p=2, dim=1)
            
            # Compute similarity matrix between anchor and target
            logits = torch.matmul(anchor_embeddings, target_embeddings.T) / temperature
            
            # Positive pairs are on the diagonal (same sample index)
            labels = torch.arange(len(anchor_embeddings), device=device)
            
            # Bidirectional loss (anchor->target and target->anchor)
            loss_forward = F.cross_entropy(logits, labels)
            loss_backward = F.cross_entropy(logits.T, labels)
            
            total_loss += (loss_forward + loss_backward) / 2
            num_modalities += 1
    
    return total_loss / num_modalities if num_modalities > 0 else torch.tensor(0.0, device=device)


def binding_modality_loss(embeddings: Dict[str, torch.Tensor], 
                         binding_modality: str, 
                         temperature: float = 0.07) -> torch.Tensor:
    """
    Alias for contrastive_loss for backward compatibility and clarity.
    
    Args:
        embeddings (Dict[str, torch.Tensor]): Dictionary of modality embeddings
        binding_modality (str): Name of the modality to use as anchor
        temperature (float): Temperature parameter for softmax
        
    Returns:
        torch.Tensor: Contrastive loss value
    """
    return contrastive_loss(embeddings, binding_modality, temperature)


def info_nce_loss(embeddings: Dict[str, torch.Tensor], 
                 binding_modality: str,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    Compute InfoNCE loss using binding modality approach.
    
    This is another name for the same binding modality contrastive loss,
    emphasizing the InfoNCE formulation.
    
    Args:
        embeddings (Dict[str, torch.Tensor]): Dictionary of modality embeddings
        binding_modality (str): Name of the modality to use as anchor
        temperature (float): Temperature parameter
        
    Returns:
        torch.Tensor: InfoNCE loss value
    """
    return contrastive_loss(embeddings, binding_modality, temperature)
