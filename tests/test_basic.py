"""
Basic tests for MultiOmicsBind core functionality.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from multiomicsbind import MultiOmicsBindWithHead, OmicsEncoder, MultiOmicsDataset


def test_omics_encoder():
    """Test OmicsEncoder functionality."""
    encoder = OmicsEncoder(input_dim=1000, embed_dim=256)
    
    # Test forward pass
    x = torch.randn(32, 1000)
    output = encoder(x)
    
    assert output.shape == (32, 256)
    assert not torch.isnan(output).any()


def test_multiomics_model():
    """Test MultiOmicsBindWithHead model."""
    input_dims = {
        'transcriptomics': 1000,
        'proteomics': 500
    }
    
    model = MultiOmicsBindWithHead(
        input_dims=input_dims,
        embed_dim=128,
        num_classes=3
    )
    
    # Test encoding
    inputs = {
        'transcriptomics': torch.randn(16, 1000),
        'proteomics': torch.randn(16, 500)
    }
    
    embeddings = model.encode(inputs)
    assert len(embeddings) == 2
    assert embeddings['transcriptomics'].shape == (16, 128)
    assert embeddings['proteomics'].shape == (16, 128)
    
    # Test classification
    logits = model(inputs)
    assert logits.shape == (16, 3)


def test_dataset_creation():
    """Test synthetic dataset creation."""
    # Create small synthetic data
    n_samples = 100
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]
    
    # Create transcriptomics data
    tx_data = np.random.randn(n_samples, 50).astype(np.float32)
    tx_df = pd.DataFrame(tx_data, columns=[f"gene_{i}" for i in range(50)])
    tx_df.insert(0, 'sample_id', sample_ids)
    tx_df.to_csv('test_tx.csv', index=False)
    
    # Create metadata
    meta_df = pd.DataFrame({
        'sample_id': sample_ids,
        'drug': np.random.choice(['A', 'B'], n_samples),
        'dose': np.random.uniform(0, 10, n_samples),
        'response': np.random.choice([0, 1, 2], n_samples)
    })
    meta_df.to_csv('test_meta.csv', index=False)
    
    # Test dataset loading
    dataset = MultiOmicsDataset(
        data_paths={'transcriptomics': 'test_tx.csv'},
        metadata_path='test_meta.csv',
        cat_cols=['drug'],
        num_cols=['dose'],
        label_col='response'
    )
    
    assert len(dataset) == n_samples
    sample = dataset[0]
    assert 'transcriptomics' in sample
    assert 'metadata' in sample
    assert 'label' in sample
    
    # Cleanup
    import os
    os.remove('test_tx.csv')
    os.remove('test_meta.csv')


def test_model_training_step():
    """Test that model can perform a training step."""
    from multiomicsbind.core.losses import contrastive_loss
    
    input_dims = {'modality1': 100, 'modality2': 200}
    model = MultiOmicsBindWithHead(input_dims, embed_dim=64)
    
    # Create sample batch
    inputs = {
        'modality1': torch.randn(8, 100),
        'modality2': torch.randn(8, 200)
    }
    
    # Forward pass
    embeddings = model(inputs)
    
    # Compute loss
    loss = contrastive_loss(embeddings)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0  # Loss should be non-negative
    
    # Test backward pass
    loss.backward()
    
    # Check that gradients exist
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__])
