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


class TemporalMultiOmicsBind(nn.Module):
    """
    Temporal Multi-Omics Binding model for mixed static and temporal data.
    
    This model extends MultiOmicsBind to handle scenarios where some modalities
    are static (single timepoint) and others are temporal (multiple timepoints).
    It uses specialized temporal encoders for time-series modalities while
    maintaining the binding modality approach for efficiency.
    
    Args:
        static_input_dims (Dict[str, int]): Input dimensions for static modalities
        temporal_input_dims (Dict[str, int]): Input dimensions for temporal modalities
        temporal_encoders (Dict[str, str]): Encoder types for temporal modalities
            Options: 'lstm', 'transformer', 'attention_pool', 'aggregation'
        binding_modality (str): Name of modality to use as binding anchor
        cat_dims (Optional[list]): Categorical metadata dimensions
        num_dims (int): Number of numerical metadata variables
        embed_dim (int): Embedding dimension
        num_classes (Optional[int]): Number of classes for classification
        dropout (float): Dropout probability
        temporal_encoder_kwargs (Optional[Dict]): Additional arguments for temporal encoders
        
    Example:
        >>> static_dims = {'transcriptomics': 6000, 'cell_painting': 1500}
        >>> temporal_dims = {'proteomics': 4000}
        >>> temporal_encoders = {'proteomics': 'lstm'}
        >>> 
        >>> model = TemporalMultiOmicsBind(
        ...     static_input_dims=static_dims,
        ...     temporal_input_dims=temporal_dims,
        ...     temporal_encoders=temporal_encoders,
        ...     binding_modality='transcriptomics',
        ...     num_classes=3
        ... )
    """
    
    def __init__(
        self,
        static_input_dims: Dict[str, int],
        temporal_input_dims: Dict[str, int],
        temporal_encoders: Dict[str, str],
        binding_modality: str,
        cat_dims: Optional[list] = None,
        num_dims: int = 0,
        embed_dim: int = 768,
        num_classes: Optional[int] = None,
        dropout: float = 0.2,
        temporal_encoder_kwargs: Optional[Dict[str, Dict]] = None
    ):
        super().__init__()
        
        # Store configuration
        self.static_input_dims = static_input_dims
        self.temporal_input_dims = temporal_input_dims
        self.temporal_encoders_config = temporal_encoders
        self.binding_modality = binding_modality
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Import temporal encoders
        from .temporal_encoders import create_temporal_encoder
        
        # Validate binding modality
        all_modalities = set(static_input_dims.keys()) | set(temporal_input_dims.keys())
        if binding_modality not in all_modalities:
            raise ValueError(f"Binding modality '{binding_modality}' not found in input modalities")
        
        # Create static encoders
        self.static_encoders = nn.ModuleDict()
        for modality, input_dim in static_input_dims.items():
            self.static_encoders[modality] = OmicsEncoder(
                input_dim=input_dim,
                embed_dim=embed_dim,
                dropout=dropout
            )
        
        # Create temporal encoders
        self.temporal_encoders = nn.ModuleDict()
        temporal_encoder_kwargs = temporal_encoder_kwargs or {}
        
        for modality, input_dim in temporal_input_dims.items():
            encoder_type = temporal_encoders[modality]
            encoder_kwargs = temporal_encoder_kwargs.get(modality, {})
            
            self.temporal_encoders[modality] = create_temporal_encoder(
                encoder_type=encoder_type,
                input_dim=input_dim,
                embed_dim=embed_dim,
                **encoder_kwargs
            )
        
        # Metadata encoder
        self.meta_encoder = None
        if cat_dims or num_dims > 0:
            self.meta_encoder = MetadataEncoder(
                cat_dims=cat_dims or [],
                num_dims=num_dims,
                embed_dim=embed_dim,
                dropout=dropout
            )
        
        # Classification head
        self.classifier = None
        if num_classes is not None:
            # Calculate total embedding dimension
            # (all modalities + metadata if present)
            total_embed_dim = len(all_modalities) * embed_dim
            if self.meta_encoder is not None:
                total_embed_dim += embed_dim
            
            self.classifier = nn.Sequential(
                nn.Linear(total_embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes)
            )
    
    def encode(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Encode all modalities into embedding space.
        
        Args:
            inputs: Dictionary containing modality data and optional metadata
            
        Returns:
            Dictionary of embeddings for each modality
        """
        embeddings = {}
        
        # Encode static modalities
        for modality in self.static_input_dims.keys():
            if modality in inputs:
                embeddings[modality] = self.static_encoders[modality](inputs[modality])
        
        # Encode temporal modalities
        for modality in self.temporal_input_dims.keys():
            if modality in inputs:
                # Get temporal data and mask
                temporal_data = inputs[modality]
                mask = inputs.get(f"{modality}_mask", None)
                
                embeddings[modality] = self.temporal_encoders[modality](
                    temporal_data, mask=mask
                )
        
        # Encode metadata if present
        if self.meta_encoder is not None and "metadata" in inputs:
            metadata_input = inputs["metadata"]
            embeddings["metadata"] = self.meta_encoder(
                x_cat=metadata_input.get("x_cat", None),
                x_num=metadata_input.get("x_num", None)
            )
        
        return embeddings
    
    def forward(
        self, 
        inputs: Dict[str, Any], 
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass through the temporal model.
        
        Args:
            inputs: Dictionary containing all input data
            return_embeddings: Whether to return embeddings along with logits
            
        Returns:
            Classification logits or (logits, embeddings) if return_embeddings=True
        """
        # Get embeddings for all modalities
        embeddings = self.encode(inputs)
        
        if self.classifier is not None:
            # Concatenate all embeddings for classification
            all_embeddings = []
            
            # Add static modality embeddings
            for modality in self.static_input_dims.keys():
                if modality in embeddings:
                    all_embeddings.append(embeddings[modality])
            
            # Add temporal modality embeddings
            for modality in self.temporal_input_dims.keys():
                if modality in embeddings:
                    all_embeddings.append(embeddings[modality])
            
            # Add metadata embedding if present
            if "metadata" in embeddings:
                all_embeddings.append(embeddings["metadata"])
            
            # Concatenate and classify
            if all_embeddings:
                combined_embedding = torch.cat(all_embeddings, dim=1)
                logits = self.classifier(combined_embedding)
                
                if return_embeddings:
                    return logits, embeddings
                return logits
            else:
                raise ValueError("No valid embeddings found for classification")
        
        else:
            # No classifier, return embeddings only
            if return_embeddings:
                return embeddings
            else:
                # Return concatenated embeddings
                all_embeddings = []
                for modality in embeddings.keys():
                    all_embeddings.append(embeddings[modality])
                
                if all_embeddings:
                    return torch.cat(all_embeddings, dim=1)
                else:
                    raise ValueError("No embeddings found")
    
    def compute_contrastive_loss(
        self, 
        embeddings: Dict[str, torch.Tensor], 
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Compute contrastive loss for temporal model.
        
        Args:
            embeddings: Dictionary of modality embeddings
            temperature: Temperature parameter for contrastive learning
            
        Returns:
            Contrastive loss value
        """
        from .losses import contrastive_loss
        return contrastive_loss(embeddings, self.binding_modality, temperature)
    
    def freeze_encoders(self):
        """Freeze all encoder parameters."""
        for encoder in self.static_encoders.values():
            for param in encoder.parameters():
                param.requires_grad = False
        
        for encoder in self.temporal_encoders.values():
            for param in encoder.parameters():
                param.requires_grad = False
        
        if self.meta_encoder is not None:
            for param in self.meta_encoder.parameters():
                param.requires_grad = False
    
    def unfreeze_encoders(self):
        """Unfreeze all encoder parameters."""
        for encoder in self.static_encoders.values():
            for param in encoder.parameters():
                param.requires_grad = True
        
        for encoder in self.temporal_encoders.values():
            for param in encoder.parameters():
                param.requires_grad = True
        
        if self.meta_encoder is not None:
            for param in self.meta_encoder.parameters():
                param.requires_grad = True
    
    def get_temporal_info(self) -> Dict[str, str]:
        """Get information about temporal encoders."""
        return self.temporal_encoders_config.copy()
    
    def set_binding_modality(self, binding_modality: str):
        """Set or change the binding modality."""
        all_modalities = set(self.static_input_dims.keys()) | set(self.temporal_input_dims.keys())
        if binding_modality not in all_modalities:
            raise ValueError(f"Binding modality '{binding_modality}' not found in modalities")
        self.binding_modality = binding_modality
    
    def get_binding_modality(self) -> str:
        """Get the current binding modality."""
        return self.binding_modality
