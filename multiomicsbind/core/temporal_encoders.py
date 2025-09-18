"""
Temporal encoders for handling time-series multi-omics data.

This module provides specialized encoders for temporal modalities that change over time,
allowing integration with static modalities measured at single timepoints.

For typical biological time series (3-20 timepoints), LSTMTemporalEncoder is recommended
as it effectively captures sequential biological processes and temporal dependencies.
Other encoders are provided for special cases and research flexibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences (used in Transformer encoder).
    """
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class LSTMTemporalEncoder(nn.Module):
    """
    LSTM-based encoder for temporal omics data (RECOMMENDED DEFAULT).
    
    This encoder processes time-series data using bidirectional LSTM
    and produces a fixed-size embedding for integration with static modalities.
    
    Best for typical biological time series with 3-20 timepoints, where sequential
    biological processes (gene regulation → protein expression → metabolite changes)
    need to be modeled with temporal dependencies.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim // (2 if bidirectional else 1),
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for temporal encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len)
        
        Returns:
            Encoded tensor of shape (batch_size, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            # Concatenate final forward and backward hidden states
            # h_n shape: (num_layers * 2, batch, hidden_size)
            forward_h = h_n[-2]  # Last layer forward
            backward_h = h_n[-1]  # Last layer backward
            final_hidden = torch.cat([forward_h, backward_h], dim=1)
        else:
            final_hidden = h_n[-1]  # Last layer
        
        # Output projection
        output = self.output_projection(final_hidden)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class TransformerTemporalEncoder(nn.Module):
    """
    Transformer-based encoder for temporal omics data (FOR SPECIAL CASES).
    
    Uses multi-head attention to capture temporal dependencies and relationships
    in time-series omics measurements.
    
    Best for long sequences (>20 timepoints) or complex multi-phase biological
    responses where attention between distant timepoints is important.
    Consider using LSTM for typical biological time series.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = None,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        if ff_dim is None:
            ff_dim = 4 * embed_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Global pooling and output projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer temporal encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)
        
        Returns:
            Encoded tensor of shape (batch_size, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        # Note: TransformerEncoder expects (seq_len, batch_size, embed_dim) for pos encoding
        x_transposed = x.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        x_transposed = self.pos_encoding(x_transposed)
        x = x_transposed.transpose(0, 1)  # Back to (batch_size, seq_len, embed_dim)
        
        # Create attention mask if provided
        attn_mask = None
        if mask is not None:
            # Convert padding mask to attention mask
            attn_mask = mask.eq(0)  # True for padding positions
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Global average pooling over sequence dimension
        # encoded shape: (batch_size, seq_len, embed_dim)
        pooled = encoded.mean(dim=1)  # Average over time steps
        
        # Output projection
        output = self.output_projection(pooled)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class AttentionPoolingTemporalEncoder(nn.Module):
    """
    Attention-pooled temporal encoder (FOR INTERPRETABILITY).
    
    Uses learned attention weights to pool across time points,
    allowing the model to focus on the most relevant timepoints.
    
    Best when you need interpretable attention weights to understand
    which timepoints are most important for predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Multi-head attention for pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable query for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Output layers
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with attention pooling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len)
        
        Returns:
            Encoded tensor of shape (batch_size, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)
        
        # Create key padding mask if provided
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask.eq(0)  # True for padding positions
        
        # Attention pooling
        pooled_output, attention_weights = self.attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )
        
        # Squeeze sequence dimension (should be 1)
        pooled_output = pooled_output.squeeze(1)
        
        # Output projection
        output = self.output_projection(pooled_output)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class TemporalAggregationEncoder(nn.Module):
    """
    Simple aggregation-based temporal encoder.
    
    Applies various aggregation strategies (mean, max, etc.) to reduce
    temporal dimension while preserving important information.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        aggregation_strategies: List[str] = ['mean', 'max', 'std'],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.strategies = aggregation_strategies
        
        # Calculate total aggregated dimension
        agg_dim = len(aggregation_strategies) * input_dim
        
        # Projection layers
        self.input_projection = nn.Linear(input_dim, input_dim)
        self.aggregation_projection = nn.Linear(agg_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with temporal aggregation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len)
        
        Returns:
            Encoded tensor of shape (batch_size, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match feature dimension
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x * mask_expanded
        
        # Apply aggregation strategies
        aggregated_features = []
        
        for strategy in self.strategies:
            if strategy == 'mean':
                if mask is not None:
                    # Masked mean
                    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                    agg = x.sum(dim=1) / valid_counts.expand(-1, x.size(-1))
                else:
                    agg = x.mean(dim=1)
            elif strategy == 'max':
                agg, _ = x.max(dim=1)
            elif strategy == 'min':
                agg, _ = x.min(dim=1)
            elif strategy == 'std':
                agg = x.std(dim=1)
            elif strategy == 'first':
                agg = x[:, 0, :]
            elif strategy == 'last':
                if mask is not None:
                    # Get last valid position for each sample
                    seq_lengths = mask.sum(dim=1) - 1  # -1 for 0-indexing
                    agg = x[torch.arange(batch_size), seq_lengths]
                else:
                    agg = x[:, -1, :]
            else:
                raise ValueError(f"Unknown aggregation strategy: {strategy}")
            
            aggregated_features.append(agg)
        
        # Concatenate all aggregated features
        aggregated = torch.cat(aggregated_features, dim=1)
        
        # Project to embedding dimension
        output = self.aggregation_projection(aggregated)
        output = F.relu(output)
        output = self.output_projection(output)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


def create_temporal_encoder(
    encoder_type: str,
    input_dim: int,
    embed_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create temporal encoders.
    
    Recommended encoder choices for biological time series:
    - 'lstm': DEFAULT - Best for typical biological time series (3-20 timepoints)
    - 'transformer': For long sequences (>20 timepoints) or complex multi-phase responses
    - 'attention_pool': For interpretable attention weights over timepoints
    - 'aggregation': For simple temporal patterns or computational efficiency
    
    Args:
        encoder_type: Type of encoder ('lstm', 'transformer', 'attention_pool', 'aggregation')
        input_dim: Input feature dimension
        embed_dim: Output embedding dimension
        **kwargs: Additional arguments for specific encoder types
    
    Returns:
        Temporal encoder instance
        
    Examples:
        >>> # Recommended for most biological omics data
        >>> encoder = create_temporal_encoder('lstm', input_dim=4000, embed_dim=256)
        
        >>> # For long time series
        >>> encoder = create_temporal_encoder('transformer', input_dim=4000, embed_dim=256)
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == 'lstm':
        return LSTMTemporalEncoder(input_dim, embed_dim, **kwargs)
    elif encoder_type == 'transformer':
        return TransformerTemporalEncoder(input_dim, embed_dim, **kwargs)
    elif encoder_type == 'attention_pool':
        return AttentionPoolingTemporalEncoder(input_dim, embed_dim, **kwargs)
    elif encoder_type == 'aggregation':
        return TemporalAggregationEncoder(input_dim, embed_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")