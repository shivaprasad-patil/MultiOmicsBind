"""
Feature importance and model interpretation utilities for MultiOmicsBind.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional


def get_gradients(
    model,
    inputs: Dict[str, torch.Tensor],
    target_class: Optional[torch.Tensor] = None
) -> Dict[str, np.ndarray]:
    """
    Compute gradients of model outputs with respect to inputs.
    
    This function enables gradient-based interpretation by computing how much
    each input feature contributes to the model's predictions. Useful for
    identifying important features driving predictions.
    
    Args:
        model: Trained TemporalMultiOmicsBind or MultiOmicsBindWithHead model
        inputs: Dictionary of input tensors
        target_class (Optional): Target class for gradient computation. If None,
            uses the predicted class
    
    Returns:
        gradients: Dictionary mapping modality names to gradient arrays
    
    Example:
        >>> from multiomicsbind import get_gradients
        >>> sample = dataset[0]
        >>> inputs = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if k != 'label'}
        >>> gradients = get_gradients(model, inputs)
        >>> # Analyze which features are most important
        >>> for modality, grad in gradients.items():
        ...     importance = np.abs(grad).mean(axis=0)
        ...     print(f"{modality}: Top feature importance = {importance.max():.4f}")
    """
    model.eval()
    
    # Enable gradients for input tensors
    for modality, data in inputs.items():
        if (
            isinstance(data, torch.Tensor) 
            and modality != 'label' 
            and torch.is_floating_point(data)
        ):
            data.requires_grad_(True)
    
    # Forward pass
    logits, _ = model(inputs, return_embeddings=True)
    
    # Use predicted class if target_class not specified
    if target_class is None:
        target_class = torch.argmax(logits, dim=1)
    
    # Compute gradients for each modality
    gradients = {}
    for modality, data in inputs.items():
        if (
            isinstance(data, torch.Tensor) 
            and modality != 'label' 
            and modality != 'metadata'
            and torch.is_floating_point(data)
        ):
            grad = torch.autograd.grad(
                outputs=logits[0, target_class[0]], 
                inputs=data,
                retain_graph=True
            )[0]
            gradients[modality] = grad.detach().cpu().numpy()
    
    return gradients


def compute_feature_importance(
    model,
    dataset,
    device,
    n_batches: int = 5,
    batch_size: int = 16,
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Compute feature importance using gradient-based analysis.
    
    This function aggregates gradients across multiple samples to identify
    which features consistently contribute to model predictions. Returns both
    raw importance scores per modality and a comprehensive DataFrame for analysis.
    
    Importance scores are normalized to [0, 1] range for each modality, where:
    - 1.0 indicates the most important feature within that modality
    - 0.0 indicates the least important feature within that modality
    - This normalization makes scores interpretable and comparable across runs
    
    Args:
        model: Trained model
        dataset: Dataset instance (TemporalMultiOmicsDataset or MultiOmicsDataset)
        device: torch device ('cuda' or 'cpu')
        n_batches (int): Number of batches to process (default: 5)
        batch_size (int): Batch size (default: 16)
        verbose (bool): Whether to print progress (default: True)
    
    Returns:
        importance_dict: Dictionary mapping modality names to importance score arrays
            of shape (n_features,). Scores are normalized to [0, 1] range per modality,
            where higher scores indicate more important features.
        importance_df: Pandas DataFrame with columns:
            - 'modality': Name of the modality
            - 'feature_index': Index of the feature
            - 'feature_name': Name of the feature
            - 'importance': Normalized importance score (0-1 range)
    
    Example:
        >>> from multiomicsbind import compute_feature_importance
        >>> importance_dict, importance_df = compute_feature_importance(
        ...     model, dataset, device, n_batches=10
        ... )
        >>> # Get top 10 most important features across all modalities
        >>> top_features = importance_df.nlargest(10, 'importance')
        >>> print(top_features[['modality', 'feature_name', 'importance']])
        >>> 
        >>> # All scores are now between 0 and 1
        >>> print(f"Min score: {importance_df['importance'].min():.3f}")  # Should be ~0.0
        >>> print(f"Max score: {importance_df['importance'].max():.3f}")  # Should be ~1.0
        >>> 
        >>> # Save for further analysis
        >>> importance_df.to_csv('feature_importance.csv', index=False)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_gradients = {}
    
    if verbose:
        print(f"Computing gradients for {n_batches} batches...")
    
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                 for k, v in batch.items()}
        gradients = get_gradients(model, inputs)
        
        for modality, grad in gradients.items():
            if modality not in all_gradients:
                all_gradients[modality] = []
            all_gradients[modality].append(grad)
        
        if verbose and (i + 1) % 2 == 0:
            print(f"  Processed batch {i+1}/{n_batches}")
    
    # Aggregate gradients
    if verbose:
        print("\nAggregating gradients...")
    
    importance_dict = {}
    importance_records = []
    
    for modality, grad_list in all_gradients.items():
        if len(grad_list) == 0:
            if verbose:
                print(f"  {modality}: Skipping (no gradients computed)")
            continue
        
        # Stack all gradients
        stacked_grads = np.vstack(grad_list)
        
        # Flatten if temporal (3D) to (n_samples, total_features)
        if stacked_grads.ndim == 3:
            if verbose:
                print(f"  {modality}: Flattening temporal gradients from {stacked_grads.shape}")
            stacked_grads = stacked_grads.reshape(stacked_grads.shape[0], -1)
            if verbose:
                print(f"  {modality}: Reshaped to {stacked_grads.shape}")
        
        # Compute importance as mean absolute gradient across samples
        importance_scores = np.mean(np.abs(stacked_grads), axis=0)
        
        # Normalize importance scores to [0, 1] range for interpretability
        # This makes scores comparable across different modalities and runs
        min_score = importance_scores.min()
        max_score = importance_scores.max()
        if max_score > min_score:  # Avoid division by zero
            importance_scores = (importance_scores - min_score) / (max_score - min_score)
        else:
            importance_scores = np.zeros_like(importance_scores)  # All features equal importance
        
        # Get feature names from dataset
        feature_names = dataset.get_feature_names(modality)
        n_features = len(feature_names)
        
        # Ensure alignment: trim or pad importance scores to match n_features
        if len(importance_scores) > n_features:
            importance_scores = importance_scores[:n_features]
            if verbose:
                print(f"  {modality}: Trimmed to {n_features} features")
        elif len(importance_scores) < n_features:
            importance_scores = np.pad(importance_scores, (0, n_features - len(importance_scores)))
            if verbose:
                print(f"  {modality}: Padded to {n_features} features")
        else:
            if verbose:
                print(f"  {modality}: {n_features} features (perfect match)")
        
        importance_dict[modality] = importance_scores
        
        # Create records for DataFrame
        for i, score in enumerate(importance_scores):
            importance_records.append({
                'modality': modality,
                'feature_index': i,
                'feature_name': feature_names[i] if i < len(feature_names) else f"{modality}_{i}",
                'importance': score
            })
    
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame(importance_records)
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    if verbose:
        print("\n✓ Feature importance computed for all modalities")
        print(f"  Total features analyzed: {len(importance_df)}")
        print(f"  Modalities: {list(importance_dict.keys())}")
    
    return importance_dict, importance_df
