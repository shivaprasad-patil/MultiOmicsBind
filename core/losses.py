"""
Loss functions for multi-omics learning.
"""

import torch
import torch.nn.functional as F
from typing import Dict


def contrastive_loss(embeddings: Dict[str, torch.Tensor], temperature: float = 0.07) -> torch.Tensor:
    """
    Compute contrastive loss between different omics modalities.
    
    This function implements a symmetric contrastive loss that encourages embeddings
    from the same sample across different modalities to be similar, while pushing
    apart embeddings from different samples.
    
    Args:
        embeddings (Dict[str, torch.Tensor]): Dictionary mapping modality names to 
            their embeddings of shape (batch_size, embed_dim)
        temperature (float): Temperature parameter for softmax (default: 0.07)
        
    Returns:
        torch.Tensor: Scalar contrastive loss value
        
    Example:
        >>> embeddings = {
        ...     'transcriptomics': torch.randn(32, 768),
        ...     'proteomics': torch.randn(32, 768)
        ... }
        >>> loss = contrastive_loss(embeddings, temperature=0.07)
    """
    if len(embeddings) < 2:
        return torch.tensor(0.0, device=next(iter(embeddings.values())).device)
    
    device = next(iter(embeddings.values())).device
    modalities = list(embeddings.keys())
    total_loss = 0.0
    num_pairs = 0

    # Normalize embeddings for cosine similarity
    normalized_embeddings = {}
    for mod, emb in embeddings.items():
        normalized_embeddings[mod] = F.normalize(emb, p=2, dim=1)

    # Compute contrastive loss for each pair of modalities
    for i, mod_q in enumerate(modalities):
        z_q = normalized_embeddings[mod_q]
        
        for j, mod_k in enumerate(modalities):
            if i != j:  # Don't compute loss for same modality
                z_k = normalized_embeddings[mod_k]
                
                # Compute similarity matrix
                logits = torch.matmul(z_q, z_k.T) / temperature
                
                # Positive pairs are on the diagonal (same sample index)
                labels = torch.arange(len(z_q), device=device)
                
                # Cross-entropy loss
                loss = F.cross_entropy(logits, labels)
                total_loss += loss
                num_pairs += 1

    return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=device)


def info_nce_loss(embeddings: Dict[str, torch.Tensor], temperature: float = 0.07) -> torch.Tensor:
    """
    Compute InfoNCE loss for multi-modal contrastive learning.
    
    Alternative implementation of contrastive loss using InfoNCE formulation.
    
    Args:
        embeddings (Dict[str, torch.Tensor]): Dictionary of modality embeddings
        temperature (float): Temperature parameter
        
    Returns:
        torch.Tensor: InfoNCE loss value
    """
    if len(embeddings) < 2:
        return torch.tensor(0.0, device=next(iter(embeddings.values())).device)
    
    device = next(iter(embeddings.values())).device
    modalities = list(embeddings.keys())
    
    # Stack all embeddings
    all_embeddings = torch.stack([embeddings[mod] for mod in modalities], dim=1)  # (batch, num_modalities, embed_dim)
    batch_size, num_modalities, embed_dim = all_embeddings.shape
    
    # Normalize embeddings
    all_embeddings = F.normalize(all_embeddings, p=2, dim=-1)
    
    total_loss = 0.0
    
    for i in range(num_modalities):
        for j in range(num_modalities):
            if i != j:
                # Query and key embeddings
                query = all_embeddings[:, i]  # (batch, embed_dim)
                key = all_embeddings[:, j]    # (batch, embed_dim)
                
                # Compute similarities
                pos_sim = torch.sum(query * key, dim=1) / temperature  # (batch,)
                
                # Negative similarities (all other samples)
                neg_sim = torch.matmul(query, key.T) / temperature  # (batch, batch)
                
                # InfoNCE loss
                logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch, batch+1)
                labels = torch.zeros(batch_size, device=device, dtype=torch.long)
                
                loss = F.cross_entropy(logits, labels)
                total_loss += loss
    
    return total_loss / (num_modalities * (num_modalities - 1))
