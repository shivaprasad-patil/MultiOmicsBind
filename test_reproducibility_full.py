"""
Comprehensive example demonstrating reproducibility in MultiOmicsBind.

This script shows that using set_seed() ensures identical results across multiple runs.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from multiomicsbind import (
    MultiOmicsDataset,
    MultiOmicsBindWithHead,
    train_multiomicsbind,
    set_seed
)

print("=" * 80)
print("MultiOmicsBind Reproducibility Example")
print("=" * 80)
print("\nThis example demonstrates that using set_seed() ensures identical results")
print("across multiple training runs with the same data.\n")

# Create synthetic data for demonstration
def create_synthetic_data(n_samples=100, seed=42):
    """Create synthetic multi-omics data"""
    np.random.seed(seed)
    
    # Create data
    data = {
        'transcriptomics': np.random.randn(n_samples, 50),
        'proteomics': np.random.randn(n_samples, 30),
    }
    
    # Create labels (binary classification)
    labels = np.random.randint(0, 2, n_samples)
    
    # Save as CSV
    pd.DataFrame(data['transcriptomics']).to_csv('/tmp/transcriptomics.csv', index=False)
    pd.DataFrame(data['proteomics']).to_csv('/tmp/proteomics.csv', index=False)
    pd.DataFrame({'label': labels}).to_csv('/tmp/metadata.csv', index=False)
    
    return data

print("📊 Creating synthetic data...")
create_synthetic_data()
print("✓ Data created\n")

# Function to train model and return final loss
def train_and_get_loss(seed_value, run_name):
    """Train model with specified seed and return final loss"""
    
    # Set seed for this run
    set_seed(seed_value, verbose=False)
    
    # Create dataset
    dataset = MultiOmicsDataset(
        data_paths={
            'transcriptomics': '/tmp/transcriptomics.csv',
            'proteomics': '/tmp/proteomics.csv'
        },
        metadata_path='/tmp/metadata.csv',
        label_col='label'
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiOmicsBindWithHead(
        input_dims=dataset.get_input_dims(),
        binding_modality='transcriptomics',
        embed_dim=32,
        num_classes=2,
        dropout=0.2
    ).to(device)
    
    # Train with seed parameter
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    trained_model = train_multiomicsbind(
        model, dataloader, optimizer, device,
        epochs=3,
        use_classification=True,
        seed=seed_value,  # Pass seed to training function
        verbose=False
    )
    
    # Get final predictions to check reproducibility
    model.eval()
    with torch.no_grad():
        sample = dataset[0]
        inputs = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if k != 'label'}
        logits, _ = model(inputs, return_embeddings=True)
        prediction = logits.cpu().numpy()[0]
    
    return prediction

# Test reproducibility
print("🔬 Testing Reproducibility")
print("=" * 80)

print("\n📌 Test 1: Same seed produces identical results")
print("-" * 80)
pred_run1 = train_and_get_loss(seed_value=42, run_name="Run 1")
pred_run2 = train_and_get_loss(seed_value=42, run_name="Run 2")

print(f"Run 1 prediction: {pred_run1}")
print(f"Run 2 prediction: {pred_run2}")
print(f"Difference: {np.abs(pred_run1 - pred_run2).max():.10f}")

identical = np.allclose(pred_run1, pred_run2, atol=1e-6)
print(f"✓ Results are identical: {identical}")

print("\n📌 Test 2: Different seeds produce different results")
print("-" * 80)
pred_seed42 = train_and_get_loss(seed_value=42, run_name="Seed 42")
pred_seed123 = train_and_get_loss(seed_value=123, run_name="Seed 123")

print(f"Seed 42 prediction:  {pred_seed42}")
print(f"Seed 123 prediction: {pred_seed123}")
print(f"Difference: {np.abs(pred_seed42 - pred_seed123).max():.10f}")

different = not np.allclose(pred_seed42, pred_seed123, atol=1e-6)
print(f"✓ Results are different: {different}")

# Summary
print("\n" + "=" * 80)
print("✅ REPRODUCIBILITY SUMMARY")
print("=" * 80)

if identical and different:
    print("🎉 MultiOmicsBind is fully reproducible!")
    print("\n📝 Key Points:")
    print("   1. Use set_seed() at the start of your script")
    print("   2. Or pass seed parameter to train_multiomicsbind()")
    print("   3. Same seed → identical results across runs")
    print("   4. Different seeds → different results (as expected)")
    
    print("\n💡 Recommended Usage:")
    print("   ```python")
    print("   from multiomicsbind import set_seed, train_multiomicsbind")
    print("   ")
    print("   # Method 1: Set seed globally")
    print("   set_seed(42)")
    print("   model = train_multiomicsbind(...)")
    print("   ")
    print("   # Method 2: Pass seed to training function")
    print("   model = train_multiomicsbind(..., seed=42)")
    print("   ```")
else:
    print("❌ Reproducibility test failed")
    print(f"   Same seed identical: {'✓' if identical else '✗'}")
    print(f"   Different seeds different: {'✓' if different else '✗'}")

print("=" * 80)
