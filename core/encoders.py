"""
Neural network encoders for different omics modalities and metadata.
"""

import torch
import torch.nn as nn


class OmicsEncoder(nn.Module):
    """
    Neural encoder for omics data (transcriptomics, proteomics, cell painting, etc.).
    
    This encoder takes high-dimensional omics data and projects it to a lower-dimensional
    embedding space using a two-layer MLP with layer normalization and dropout.
    
    Args:
        input_dim (int): Dimensionality of input omics data
        embed_dim (int): Dimensionality of output embedding (default: 768)
        dropout (float): Dropout probability (default: 0.2)
    """
    
    def __init__(self, input_dim: int, embed_dim: int = 768, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the omics encoder.
        
        Args:
            x (torch.Tensor): Input omics data of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Encoded embeddings of shape (batch_size, embed_dim)
        """
        return self.encoder(x)


class MetadataEncoder(nn.Module):
    """
    Neural encoder for metadata containing categorical and numerical features.
    
    This encoder handles mixed-type metadata by using embedding layers for categorical
    variables and linear layers for numerical variables, then combining them.
    
    Args:
        cat_dims (list): List of cardinalities for each categorical variable
        num_dims (int): Number of numerical variables
        embed_dim (int): Dimensionality of output embedding (default: 768)
        dropout (float): Dropout probability (default: 0.2)
    """
    
    def __init__(self, cat_dims: list, num_dims: int, embed_dim: int = 768, dropout: float = 0.2):
        super().__init__()
        self.cat_dims = cat_dims
        self.num_dims = num_dims
        self.embed_dim = embed_dim
        
        # Embedding layers for categorical variables
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, embed_dim) for cat_dim in cat_dims
        ])
        
        # Linear layer for numerical variables
        self.num_encoder = nn.Linear(num_dims, embed_dim) if num_dims > 0 else None
        
        # Projection layer to combine all embeddings
        total_inputs = len(cat_dims) + (1 if num_dims > 0 else 0)
        if total_inputs > 0:
            self.project = nn.Sequential(
                nn.Linear(embed_dim * total_inputs, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.project = nn.Identity()

    def forward(self, x_cat: torch.Tensor = None, x_num: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the metadata encoder.
        
        Args:
            x_cat (torch.Tensor, optional): Categorical features of shape (batch_size, num_cat_features)
            x_num (torch.Tensor, optional): Numerical features of shape (batch_size, num_num_features)
            
        Returns:
            torch.Tensor: Encoded metadata embeddings of shape (batch_size, embed_dim)
        """
        embeddings = []
        
        # Process categorical features
        if x_cat is not None:
            for i, emb_layer in enumerate(self.cat_embeddings):
                embeddings.append(emb_layer(x_cat[:, i]))
        
        # Process numerical features
        if x_num is not None and self.num_encoder:
            embeddings.append(self.num_encoder(x_num))
        
        if not embeddings:
            raise ValueError("At least one of x_cat or x_num must be provided")
            
        # Combine all embeddings
        x = torch.cat(embeddings, dim=-1)
        return self.project(x)
