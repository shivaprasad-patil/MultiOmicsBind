"""
Utility functions for data preprocessing and handling in MultiOmicsBind.
"""

import numpy as np
from typing import Union


def fix_nan_values(
    dataset,
    modality: str = 'proteomics',
    fill_value: float = 0.0,
    verbose: bool = True
):
    """
    Replace NaN values in temporal or static data.
    
    This function handles NaN values that may appear in omics data, particularly
    in temporal proteomics measurements. It replaces NaN values while preserving
    the masking information (if available) that indicates valid timepoints.
    
    Args:
        dataset: TemporalMultiOmicsDataset or MultiOmicsDataset instance
        modality (str): Name of the modality to fix (default: 'proteomics')
        fill_value (float): Value to replace NaN with (default: 0.0)
        verbose (bool): Whether to print fix information (default: True)
    
    Returns:
        dataset: Modified dataset with NaN values replaced
    
    Example:
        >>> from multiomicsbind import TemporalMultiOmicsDataset, fix_nan_values
        >>> dataset = TemporalMultiOmicsDataset(...)
        >>> # Check for NaN values
        >>> sample = dataset[0]
        >>> has_nan = torch.isnan(sample['proteomics']).any()
        >>> print(f"Has NaN before fix: {has_nan}")
        >>> 
        >>> # Fix NaN values
        >>> dataset = fix_nan_values(dataset, modality='proteomics')
        >>> 
        >>> # Verify fix
        >>> sample = dataset[0]
        >>> has_nan = torch.isnan(sample['proteomics']).any()
        >>> print(f"Has NaN after fix: {has_nan}")
    
    Note:
        The function modifies the dataset in-place and returns it for convenience.
        For temporal data, the mask (e.g., 'proteomics_mask') is preserved and
        will still correctly indicate valid timepoints despite the NaN replacement.
    """
    if verbose:
        print(f"Fixing NaN values in {modality} data...")
    
    # Try to access temporal data first
    if hasattr(dataset, 'temporal_data') and modality in dataset.temporal_data:
        data_dict = dataset.temporal_data
        data_type = 'temporal'
    elif hasattr(dataset, 'static_data') and modality in dataset.static_data:
        data_dict = dataset.static_data
        data_type = 'static'
    elif hasattr(dataset, 'data') and modality in dataset.data:
        data_dict = dataset.data
        data_type = 'general'
    else:
        if verbose:
            print(f"  ⚠ Warning: Modality '{modality}' not found in dataset")
        return dataset
    
    # Get the data array
    data_array = data_dict[modality]
    
    if isinstance(data_array, np.ndarray):
        nan_before = np.isnan(data_array).sum()
        
        if nan_before > 0:
            # Replace NaN values
            data_dict[modality] = np.nan_to_num(data_array, nan=fill_value)
            nan_after = np.isnan(data_dict[modality]).sum()
            
            if verbose:
                total_values = data_array.size
                pct_nan = (nan_before / total_values) * 100
                print(f"  ✓ Replaced {nan_before:,} NaN values ({pct_nan:.2f}%) in {modality} ({data_type})")
                print(f"    Remaining NaN values: {nan_after}")
                
                if nan_after == 0:
                    print(f"  ✓ All NaN values successfully replaced with {fill_value}")
        else:
            if verbose:
                print(f"  ✓ No NaN values found in {modality}")
    else:
        if verbose:
            print(f"  ⚠ Warning: Data for {modality} is not a numpy array (type: {type(data_array)})")
    
    # Verify fix by checking a few samples
    if verbose:
        print(f"\nVerifying fix by checking samples...")
        new_nan_count = 0
        n_samples_to_check = min(10, len(dataset))
        
        for i in range(n_samples_to_check):
            sample = dataset[i]
            if modality in sample:
                import torch
                if isinstance(sample[modality], torch.Tensor):
                    new_nan_count += torch.isnan(sample[modality]).sum().item()
        
        print(f"  NaN count in first {n_samples_to_check} samples: {new_nan_count}")
        if new_nan_count == 0:
            print(f"  ✓ Verification passed: No NaN values in samples")
        else:
            print(f"  ⚠ Warning: {new_nan_count} NaN values still present in samples")
    
    return dataset


def check_nan_values(dataset, verbose: bool = True):
    """
    Check for NaN values across all modalities in the dataset.
    
    Performs a comprehensive check for NaN values in all data modalities,
    reporting statistics for each modality found.
    
    Args:
        dataset: TemporalMultiOmicsDataset or MultiOmicsDataset instance
        verbose (bool): Whether to print detailed information (default: True)
    
    Returns:
        nan_stats: Dictionary mapping modality names to NaN statistics:
            - 'total_nans': Total number of NaN values
            - 'samples_with_nan': Number of samples containing NaN
            - 'percentage': Percentage of values that are NaN
    
    Example:
        >>> from multiomicsbind import check_nan_values
        >>> nan_stats = check_nan_values(dataset)
        >>> for modality, stats in nan_stats.items():
        ...     if stats['total_nans'] > 0:
        ...         print(f"{modality}: {stats['percentage']:.2f}% NaN values")
    """
    import torch
    
    if verbose:
        print("Checking dataset for NaN values...\n")
    
    nan_stats = {}
    n_samples_to_check = min(20, len(dataset))
    
    # Check first sample to identify modalities
    sample = dataset[0]
    modalities = [key for key, value in sample.items() 
                 if isinstance(value, torch.Tensor) and key != 'label']
    
    # Check multiple samples
    for modality in modalities:
        nan_stats[modality] = {
            'total_nans': 0,
            'samples_with_nan': 0,
            'total_values': 0
        }
    
    for i in range(n_samples_to_check):
        sample = dataset[i]
        for modality in modalities:
            if modality in sample and isinstance(sample[modality], torch.Tensor):
                tensor = sample[modality]
                nan_mask = torch.isnan(tensor)
                
                if nan_mask.any():
                    nan_stats[modality]['samples_with_nan'] += 1
                    nan_stats[modality]['total_nans'] += nan_mask.sum().item()
                
                nan_stats[modality]['total_values'] += tensor.numel()
    
    # Calculate percentages
    for modality, stats in nan_stats.items():
        if stats['total_values'] > 0:
            stats['percentage'] = (stats['total_nans'] / stats['total_values']) * 100
        else:
            stats['percentage'] = 0.0
    
    # Print results
    if verbose:
        print(f"NaN Statistics (checked {n_samples_to_check} samples):")
        print("=" * 70)
        for modality, stats in nan_stats.items():
            if stats['total_nans'] > 0:
                print(f"{modality}:")
                print(f"  Samples with NaN: {stats['samples_with_nan']}/{n_samples_to_check}")
                print(f"  Total NaN values: {stats['total_nans']:,}")
                print(f"  Percentage: {stats['percentage']:.2f}%")
            else:
                print(f"{modality}: ✓ No NaN values")
        print("=" * 70)
    
    return nan_stats
