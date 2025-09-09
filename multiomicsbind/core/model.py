"""
Main MultiOmicsBind model implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any

from .encoders import OmicsEncoder, MetadataEncoder


class MultiOmicsBindWithHead(nn.Module):
    """
    Multi-Omics Binding model with optional classification head.
    
    This model integrates multiple omics modalities (transcriptomics, proteomics, 
    cell painting, etc.) and optional metadata into a unified embedding space.
    It supports both contrastive learning and supervised classification.
    
    Args:
        input_dims (Dict[str, int]): Dictionary mapping modality names to input dimensions
        binding_modality (str): Name of modality to use as binding anchor (required)
        cat_dims (Optional[list]): List of cardinalities for categorical metadata variables
        num_dims (int): Number of numerical metadata variables (default: 0)
        embed_dim (int): Dimensionality of embedding space (default: 768)
        num_classes (Optional[int]): Number of classes for classification (None for no classification)
        dropout (float): Dropout probability (default: 0.2)
        
    Example:
        >>> input_dims = {
        ...     'transcriptomics': 6000,
        ...     'proteomics': 4000,
        ...     'cell_painting': 1500
        ... }
        >>> model = MultiOmicsBindWithHead(
        ...     input_dims=input_dims,
        ...     binding_modality='transcriptomics',  # Use transcriptomics as anchor
        ...     cat_dims=[10, 5],  # 10 drug types, 5 cell lines
        ...     num_dims=1,       # dose
        ...     num_classes=3     # 3 response classes
        ... )
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        binding_modality: str,
        cat_dims: Optional[list] = None,
        num_dims: int = 0,
        embed_dim: int = 768,
        num_classes: Optional[int] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.binding_modality = binding_modality
        
        # Validate binding modality (required)
        if binding_modality not in input_dims:
            raise ValueError(f"Binding modality '{binding_modality}' not found in input_dims. "
                           f"Available modalities: {list(input_dims.keys())}")
        
        # Omics encoders for each modality
        self.encoders = nn.ModuleDict({
            modality: OmicsEncoder(dim, embed_dim, dropout) 
            for modality, dim in input_dims.items()
        })

        # Metadata encoder (optional)
        self.meta_encoder = None
        if cat_dims or num_dims > 0:
            self.meta_encoder = MetadataEncoder(
                cat_dims or [], num_dims, embed_dim, dropout
            )

        # Classification head (optional)
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 1.5),  # Higher dropout for classifier
                nn.Linear(embed_dim // 2, num_classes)
            )

    def encode(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Encode all input modalities into embedding space.
        
        Args:
            inputs (Dict[str, Any]): Dictionary containing modality data and optional metadata
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping modality names to embeddings
        """
        embeddings = {}

        # Encode omics modalities
        for modality, data in inputs.items():
            if modality != "metadata" and modality != "label" and data is not None:
                if modality in self.encoders:
                    embeddings[modality] = self.encoders[modality](data)

        # Encode metadata if present
        if "metadata" in inputs and self.meta_encoder is not None:
            meta_data = inputs["metadata"]
            if isinstance(meta_data, dict):
                meta_embedding = self.meta_encoder(**meta_data)
                embeddings["metadata"] = meta_embedding

        return embeddings

    def forward(
        self, 
        inputs: Dict[str, Any], 
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], tuple]:
        """
        Forward pass of the model.
        
        Args:
            inputs (Dict[str, Any]): Input data dictionary
            return_embeddings (bool): Whether to return individual embeddings
            
        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor], tuple]: 
                - If classifier is None: embeddings dictionary
                - If classifier exists and return_embeddings=False: logits
                - If classifier exists and return_embeddings=True: (logits, embeddings)
        """
        embeddings = self.encode(inputs)

        # If no classifier, return embeddings for contrastive learning
        if self.classifier is None:
            return embeddings

        # Combine embeddings for classification
        if len(embeddings) > 0:
            combined = torch.mean(torch.stack(list(embeddings.values())), dim=0)
            logits = self.classifier(combined)
            
            if return_embeddings:
                return logits, embeddings
            else:
                return logits
        else:
            # No valid embeddings found
            batch_size = next(iter(inputs.values())).shape[0]
            device = next(iter(inputs.values())).device
            dummy_logits = torch.zeros(batch_size, self.num_classes, device=device)
            
            if return_embeddings:
                return dummy_logits, embeddings
            else:
                return dummy_logits

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embed_dim
    
    def get_modalities(self) -> list:
        """Get list of supported modalities."""
        modalities = list(self.input_dims.keys())
        if self.meta_encoder is not None:
            modalities.append("metadata")
        return modalities
    
    def freeze_encoders(self):
        """Freeze encoder parameters (useful for fine-tuning classifier only)."""
        for encoder in self.encoders.values():
            for param in encoder.parameters():
                param.requires_grad = False
        
        if self.meta_encoder is not None:
            for param in self.meta_encoder.parameters():
                param.requires_grad = False
    
    def unfreeze_encoders(self):
        """Unfreeze encoder parameters."""
        for encoder in self.encoders.values():
            for param in encoder.parameters():
                param.requires_grad = True
        
        if self.meta_encoder is not None:
            for param in self.meta_encoder.parameters():
                param.requires_grad = True
    
    def compute_contrastive_loss(self, embeddings: Dict[str, torch.Tensor], 
                               temperature: float = 0.07) -> torch.Tensor:
        """
        Compute contrastive loss using the model's binding modality configuration.
        
        Args:
            embeddings (Dict[str, torch.Tensor]): Dictionary of modality embeddings
            temperature (float): Temperature parameter for contrastive learning
            
        Returns:
            torch.Tensor: Contrastive loss value
        """
        from .losses import contrastive_loss
        return contrastive_loss(embeddings, self.binding_modality, temperature)
    
    def set_binding_modality(self, binding_modality: str):
        """
        Set or change the binding modality.
        
        Args:
            binding_modality (str): Name of modality to use as binding anchor (required)
        """
        if binding_modality not in self.input_dims:
            raise ValueError(f"Binding modality '{binding_modality}' not found in input_dims. "
                           f"Available modalities: {list(self.input_dims.keys())}")
        self.binding_modality = binding_modality
    
    def get_binding_modality(self) -> str:
        """Get the current binding modality."""
        return self.binding_modality
