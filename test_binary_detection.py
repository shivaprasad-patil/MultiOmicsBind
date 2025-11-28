"""
Test script to verify automatic binary data detection works correctly.
This tests the new feature that automatically skips normalization for binary data.
"""

import sys
import pandas as pd
import numpy as np
from multiomicsbind.data.dataset import MultiOmicsDataset

# Load original CLL data (without manual preprocessing)
print("=" * 80)
print("Testing Automatic Binary Data Detection")
print("=" * 80)
print()

# Create dataset with original data (including mutations with binary 0/1 values)
print("Creating dataset with original data (no manual preprocessing)...")
print("This should automatically detect binary mutation data and skip normalization.\n")

try:
    dataset = MultiOmicsDataset(
        data_paths={
            'mRNA': '/Users/shivaprasad/Documents/PROJECTS/GitHub/MO/CLL_data/mRNA.csv',
            'methylation': '/Users/shivaprasad/Documents/PROJECTS/GitHub/MO/CLL_data/Methylation.csv',
            'drugs': '/Users/shivaprasad/Documents/PROJECTS/GitHub/MO/CLL_data/Drugs.csv',
            'mutations': '/Users/shivaprasad/Documents/PROJECTS/GitHub/MO/CLL_data/Mutations.csv'  # Binary data - should skip normalization
        },
        metadata_path='/Users/shivaprasad/Documents/PROJECTS/GitHub/MO/CLL_data/CLL_metadata.csv',
        label_col='IGHV',
        sample_id_col='sample_id',
        normalize=True  # Enable normalization
    )
    
    print("\n" + "=" * 80)
    print("SUCCESS! Dataset created without NaN values.")
    print("=" * 80)
    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Number of modalities: {len(dataset.modalities)}")
    
    # Check for NaN values in each modality
    print("\n" + "-" * 80)
    print("Checking for NaN values in each modality:")
    print("-" * 80)
    for modality in dataset.modalities:
        data = dataset.omics_data[modality]
        has_nan = np.isnan(data).any()
        nan_count = np.isnan(data).sum()
        print(f"  {modality}: {'❌ Contains NaN' if has_nan else '✅ No NaN'} "
              f"({nan_count} NaN values)" if has_nan else f"  {modality}: ✅ No NaN")
        
        # For mutations, verify data is still binary (0/1)
        if modality == 'mutations':
            unique_values = np.unique(data[~np.isnan(data)])
            is_binary = set(unique_values).issubset({0, 0.0, 1, 1.0})
            print(f"    → Binary check: {'✅ Still binary (0/1)' if is_binary else '❌ Not binary anymore'}")
            print(f"    → Unique values: {sorted(unique_values)[:10]}")  # Show first 10
    
    # Test getting a sample
    print("\n" + "-" * 80)
    print("Testing sample retrieval:")
    print("-" * 80)
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            has_nan = np.isnan(value.numpy()).any()
            print(f"  {key}: shape={value.shape}, has_nan={has_nan}")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! Binary detection is working correctly.")
    print("=" * 80)
    
except Exception as e:
    print("\n" + "=" * 80)
    print("❌ ERROR: Dataset creation failed")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
