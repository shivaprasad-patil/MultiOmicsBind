"""
Utility functions for data preprocessing and handling in MultiOmicsBind.
"""

import numpy as np
import random
import torch
from typing import Union, Optional


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


def check_and_fix_all_nan_values(
    dataset,
    fill_value: float = 0.0,
    verbose: bool = True
):
    """
    Automatically check and fix NaN values across ALL modalities in the dataset.
    
    This convenience function scans all modalities (both static and temporal) for
    NaN values and fixes them automatically. It's the recommended way to ensure
    clean data before training.
    
    Args:
        dataset: TemporalMultiOmicsDataset or MultiOmicsDataset instance
        fill_value (float): Value to replace NaN with (default: 0.0)
        verbose (bool): Whether to print detailed information (default: True)
    
    Returns:
        dataset: Modified dataset with all NaN values replaced
        nan_summary: Dictionary with NaN statistics for each modality before fixing
    
    Example:
        >>> from multiomicsbind import TemporalMultiOmicsDataset, check_and_fix_all_nan_values
        >>> 
        >>> # Load dataset
        >>> dataset = TemporalMultiOmicsDataset(...)
        >>> 
        >>> # Check and fix ALL modalities automatically (one line!)
        >>> dataset, nan_summary = check_and_fix_all_nan_values(dataset, verbose=True)
        >>> 
        >>> # Dataset is now ready for training
        >>> model, history = train_temporal_model(dataset, ...)
    
    Note:
        This function is recommended over calling fix_nan_values() manually for
        each modality, as it ensures no modality is accidentally skipped.
    """
    if verbose:
        print("=" * 80)
        print("CHECKING AND FIXING NaN VALUES ACROSS ALL MODALITIES")
        print("=" * 80)
    
    # Collect all modalities
    all_modalities = []
    
    # Get static modalities
    if hasattr(dataset, 'static_data'):
        all_modalities.extend(dataset.static_data.keys())
    
    # Get temporal modalities
    if hasattr(dataset, 'temporal_data'):
        all_modalities.extend(dataset.temporal_data.keys())
    
    # Get general data modalities (for non-temporal datasets)
    if hasattr(dataset, 'data') and not hasattr(dataset, 'static_data'):
        all_modalities.extend(dataset.data.keys())
    
    if verbose:
        print(f"\nFound {len(all_modalities)} modalities: {all_modalities}")
        print()
    
    # Check for NaN values in each modality
    nan_summary = {}
    modalities_with_nan = []
    
    for modality in all_modalities:
        # Get the data array
        if hasattr(dataset, 'temporal_data') and modality in dataset.temporal_data:
            data_array = dataset.temporal_data[modality]
            data_type = 'temporal'
        elif hasattr(dataset, 'static_data') and modality in dataset.static_data:
            data_array = dataset.static_data[modality]
            data_type = 'static'
        elif hasattr(dataset, 'data') and modality in dataset.data:
            data_array = dataset.data[modality]
            data_type = 'general'
        else:
            continue
        
        if isinstance(data_array, np.ndarray):
            nan_count = np.isnan(data_array).sum()
            total_values = data_array.size
            
            nan_summary[modality] = {
                'nan_count': int(nan_count),
                'total_values': int(total_values),
                'percentage': (nan_count / total_values * 100) if total_values > 0 else 0.0,
                'data_type': data_type
            }
            
            if nan_count > 0:
                modalities_with_nan.append(modality)
    
    # Print summary
    if verbose:
        print("NaN Detection Summary:")
        print("-" * 80)
        for modality, stats in nan_summary.items():
            if stats['nan_count'] > 0:
                print(f"⚠ {modality} ({stats['data_type']}): {stats['nan_count']:,} NaN values "
                      f"({stats['percentage']:.2f}% of {stats['total_values']:,} total)")
            else:
                print(f"✓ {modality} ({stats['data_type']}): No NaN values")
        print("-" * 80)
    
    # Fix NaN values in modalities that have them
    if modalities_with_nan:
        if verbose:
            print(f"\nFixing NaN values in {len(modalities_with_nan)} modalities...")
            print()
        
        for modality in modalities_with_nan:
            dataset = fix_nan_values(
                dataset, 
                modality=modality, 
                fill_value=fill_value, 
                verbose=verbose
            )
            if verbose:
                print()
        
        if verbose:
            print("=" * 80)
            print("✓ ALL NaN VALUES FIXED")
            print("=" * 80)
    else:
        if verbose:
            print("\n✓ No NaN values found in any modality!")
            print("=" * 80)
    
    return dataset, nan_summary


def set_seed(seed: int = 42, deterministic: bool = True, verbose: bool = True):
    """
    Set random seeds for reproducibility across numpy, random, and PyTorch.
    
    This function ensures that running the same code multiple times with the same
    data produces identical results. It sets seeds for:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's random number generator (CPU and CUDA)
    - PyTorch's deterministic algorithms (optional)
    
    Args:
        seed (int): Random seed value (default: 42)
        deterministic (bool): Whether to use deterministic algorithms in PyTorch.
            This may reduce performance but ensures complete reproducibility.
            (default: True)
        verbose (bool): Whether to print confirmation message (default: True)
    
    Returns:
        None
    
    Example:
        >>> from multiomicsbind import set_seed
        >>> 
        >>> # Set seed at the beginning of your script
        >>> set_seed(42)
        >>> 
        >>> # Now all random operations will be reproducible
        >>> model = MultiOmicsBindWithHead(...)
        >>> dataset = MultiOmicsDataset(...)
        >>> train_multiomicsbind(model, dataloader, ...)
        >>> 
        >>> # Running the same code again with seed=42 will give identical results
    
    Note:
        - For complete reproducibility, call this function before any random operations
        - deterministic=True may impact performance in some PyTorch operations
        - CUDA operations may still have minor variations due to hardware/driver differences
        - For maximum reproducibility on GPU, also set CUBLAS_WORKSPACE_CONFIG=:4096:8
          as an environment variable before running your script
    
    Reference:
        PyTorch reproducibility guide: https://pytorch.org/docs/stable/notes/randomness.html
    """
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed for CPU
    torch.manual_seed(seed)
    
    # Set PyTorch random seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Configure deterministic behavior
    if deterministic:
        # Make cuDNN deterministic (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Use deterministic algorithms in PyTorch (PyTorch 1.8+)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # For older PyTorch versions
            torch.set_deterministic(True)
    
    if verbose:
        print(f"✓ Random seed set to {seed} for reproducibility")
        if deterministic:
            print(f"  • Deterministic mode: ON (may reduce performance)")
        else:
            print(f"  • Deterministic mode: OFF (better performance, less strict reproducibility)")
        if torch.cuda.is_available():
            print(f"  • CUDA seeds set for {torch.cuda.device_count()} GPU(s)")
