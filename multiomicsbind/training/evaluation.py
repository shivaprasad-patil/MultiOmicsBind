"""
Evaluation utilities for MultiOmicsBind models.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple


def evaluate_temporal_model(
    model,
    dataset,
    device,
    batch_size: int = 32
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Evaluate model and extract embeddings, labels, and predictions.
    
    This function performs a full evaluation pass over the dataset, extracting
    embeddings for each modality, true labels, and model predictions. Useful for
    downstream analysis like visualization, feature importance, and performance metrics.
    
    ⚠️ IMPORTANT - AVOIDING DATA LEAKAGE:
        Pass a HELD-OUT TEST SET or VALIDATION SET, not the training set!
        
        Example (CORRECT):
            train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])
            model, history = train_temporal_model(train_dataset, ...)
            embeddings, labels, preds = evaluate_temporal_model(model, test_dataset, ...)
        
        Example (WRONG):
            model, history = train_temporal_model(dataset, ...)
            embeddings, labels, preds = evaluate_temporal_model(model, dataset, ...)
    
    Args:
        model: Trained TemporalMultiOmicsBind or MultiOmicsBindWithHead model
        dataset: Test or validation dataset (NOT training set!) for evaluation.
                Can be either a Dataset object or a Subset from random_split().
        device: torch device ('cuda' or 'cpu')
        batch_size (int): Batch size for evaluation (default: 32)
    
    Returns:
        embeddings_dict: Dictionary mapping modality names to embedding arrays
            of shape (n_samples, embed_dim)
        labels: Array of true labels of shape (n_samples,)
        predictions: Array of predicted labels of shape (n_samples,)
    
    Example:
        >>> from multiomicsbind import evaluate_temporal_model
        >>> embeddings, labels, predictions = evaluate_temporal_model(
        ...     model, test_dataset, device  # Use test_dataset, not full dataset!
        ... )
        >>> print(f"Embeddings for modalities: {list(embeddings.keys())}")
        >>> print(f"Test Accuracy: {(predictions == labels).mean():.4f}")
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_embeddings = {}
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                     for k, v in batch.items()}
            labels = inputs.pop('label').to(device)
            
            logits, embeddings = model(inputs, return_embeddings=True)
            predictions = logits.argmax(1)
            
            # Store embeddings for each modality (initialize on first batch)
            for modality, emb in embeddings.items():
                if modality not in all_embeddings:
                    all_embeddings[modality] = []
                all_embeddings[modality].append(emb.cpu().numpy())
            
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    
    # Concatenate all batches
    embeddings_dict = {mod: np.vstack(emb_list) for mod, emb_list in all_embeddings.items()}
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    
    return embeddings_dict, labels, predictions


def compute_cross_modal_similarity(
    embeddings_dict: Dict[str, np.ndarray],
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute pairwise cosine similarity between embeddings from different modalities.
    
    This function calculates the cosine similarity between all pairs of modalities
    to understand how well different data types align in the learned embedding space.
    High similarity indicates the model has learned shared representations across modalities.
    
    Args:
        embeddings_dict: Dictionary mapping modality names to embedding arrays
            of shape (n_samples, embed_dim)
        verbose (bool): Whether to print similarity statistics (default: True)
    
    Returns:
        similarity_matrices: Dictionary mapping comparison names (e.g., 'mod1_vs_mod2')
            to similarity matrices of shape (n_samples, n_samples)
    
    Example:
        >>> from multiomicsbind import compute_cross_modal_similarity
        >>> similarity_matrices = compute_cross_modal_similarity(embeddings)
        >>> # Analyze how well transcriptomics and proteomics align
        >>> tx_pr_sim = similarity_matrices['transcriptomics_vs_proteomics']
        >>> print(f"Mean cross-modal similarity: {tx_pr_sim.mean():.4f}")
    """
    modalities = list(embeddings_dict.keys())
    similarity_matrices = {}
    
    for i, mod1 in enumerate(modalities):
        for mod2 in modalities[i+1:]:
            # Check for NaN values in embeddings
            emb1 = embeddings_dict[mod1]
            emb2 = embeddings_dict[mod2]
            
            if np.isnan(emb1).any():
                if verbose:
                    nan_count = np.isnan(emb1).sum()
                    print(f"⚠ Warning: {mod1} embeddings contain {nan_count} NaN values - replacing with 0")
                emb1 = np.nan_to_num(emb1, nan=0.0)
            
            if np.isnan(emb2).any():
                if verbose:
                    nan_count = np.isnan(emb2).sum()
                    print(f"⚠ Warning: {mod2} embeddings contain {nan_count} NaN values - replacing with 0")
                emb2 = np.nan_to_num(emb2, nan=0.0)
            
            # Compute cosine similarity
            sim_matrix = cosine_similarity(emb1, emb2)
            similarity_matrices[f"{mod1}_vs_{mod2}"] = sim_matrix
            
            # Compute mean similarity
            if verbose:
                mean_sim = np.mean(sim_matrix)
                std_sim = np.std(sim_matrix)
                print(f"{mod1} vs {mod2}: Mean similarity = {mean_sim:.4f} ± {std_sim:.4f}")
    
    return similarity_matrices


def analyze_similarity_by_class(
    similarity_matrices: Dict[str, np.ndarray],
    labels: np.ndarray,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Analyze cross-modal similarity patterns by response class.
    
    Computes per-class statistics for cross-modal similarities to understand
    if certain classes have better cross-modal alignment than others.
    
    Args:
        similarity_matrices: Dictionary of similarity matrices from compute_cross_modal_similarity
        labels: Array of class labels of shape (n_samples,)
        verbose (bool): Whether to print analysis results (default: True)
    
    Returns:
        results_dict: Dictionary mapping comparison names to dictionaries containing:
            - 'per_class_mean': Mean similarity per class
            - 'per_class_std': Standard deviation per class
            - 'overall_mean': Overall mean similarity
            - 'overall_std': Overall standard deviation
    
    Example:
        >>> from multiomicsbind import analyze_similarity_by_class
        >>> results = analyze_similarity_by_class(similarity_matrices, labels)
        >>> # Check if responders have better cross-modal alignment
        >>> for comparison, stats in results.items():
        ...     print(f"{comparison}:")
        ...     print(f"  Class 0: {stats['per_class_mean'][0]:.4f}")
        ...     print(f"  Class 1: {stats['per_class_mean'][1]:.4f}")
    """
    unique_labels = np.unique(labels)
    results_dict = {}
    
    for comparison, sim_matrix in similarity_matrices.items():
        # Get diagonal (self-similarity)
        diagonal_sim = np.diag(sim_matrix)
        
        # Compute statistics per class
        per_class_mean = {}
        per_class_std = {}
        
        if verbose:
            print(f"\n{comparison}:")
        
        for label in unique_labels:
            mask = labels == label
            class_sim = diagonal_sim[mask]
            per_class_mean[int(label)] = np.mean(class_sim)
            per_class_std[int(label)] = np.std(class_sim)
            
            if verbose:
                print(f"  Class {label}: Mean self-similarity = "
                      f"{per_class_mean[int(label)]:.4f} ± {per_class_std[int(label)]:.4f}")
        
        results_dict[comparison] = {
            'per_class_mean': per_class_mean,
            'per_class_std': per_class_std,
            'overall_mean': np.mean(diagonal_sim),
            'overall_std': np.std(diagonal_sim)
        }
    
    return results_dict
