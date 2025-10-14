"""
High-level analysis workflows for MultiOmicsBind.
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.metrics import classification_report, confusion_matrix

from .training.evaluation import (
    evaluate_temporal_model, 
    compute_cross_modal_similarity,
    analyze_similarity_by_class
)
from .training.interpretation import compute_feature_importance
from .utils.visualization import (
    plot_embeddings_umap,
    plot_confusion_matrix,
    plot_training_history_detailed,
    plot_cross_modal_similarity_matrices,
    plot_feature_importance_distribution
)


def create_analysis_report(
    model,
    dataset,
    device,
    history: Optional[Dict[str, list]] = None,
    output_dir: str = './results',
    compute_importance: bool = True,
    compute_similarity: bool = True,
    n_importance_batches: int = 5,
    top_k_features: int = 30,
    class_names: Optional[list] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate comprehensive analysis report with all visualizations and metrics.
    
    This high-level function performs a complete analysis of a trained model,
    including performance evaluation, embedding visualization, feature importance
    analysis, and cross-modal similarity computation. All results are saved to
    the specified output directory.
    
    ⚠️ IMPORTANT - AVOIDING DATA LEAKAGE:
        Always pass a HELD-OUT TEST SET or VALIDATION SET to this function,
        NOT the training set! Evaluating on training data gives misleading
        results due to overfitting and memorization.
        
        Example (CORRECT):
            train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])
            model, history = train_temporal_model(train_dataset, ...)
            report = create_analysis_report(model, test_dataset, ...)  # ✓ Test set!
        
        Example (WRONG - DATA LEAKAGE):
            model, history = train_temporal_model(dataset, ...)
            report = create_analysis_report(model, dataset, ...)  # ✗ Same data!
    
    Args:
        model: Trained TemporalMultiOmicsBind or MultiOmicsBindWithHead model
        dataset: Test or validation dataset (NOT the training set!) for evaluation
        dataset: Dataset instance (TemporalMultiOmicsDataset or MultiOmicsDataset)
        device: torch device ('cuda' or 'cpu')
        history (Optional): Training history dictionary. If provided, will plot training curves.
        output_dir (str): Directory to save all results (default: './results')
        compute_importance (bool): Whether to compute feature importance (default: True)
        compute_similarity (bool): Whether to compute cross-modal similarity (default: True)
        n_importance_batches (int): Number of batches for importance calculation (default: 5)
        top_k_features (int): Number of top features to visualize (default: 30)
        class_names (Optional): List of class names for confusion matrix. If None,
            uses class indices.
        verbose (bool): Whether to print progress information (default: True)
    
    Returns:
        report_dict: Dictionary containing all analysis results:
            - 'embeddings': Dictionary of embeddings per modality
            - 'labels': Array of true labels
            - 'predictions': Array of predicted labels
            - 'classification_report': Text classification report
            - 'confusion_matrix': Confusion matrix array
            - 'importance_df': Feature importance DataFrame (if compute_importance=True)
            - 'similarity_matrices': Cross-modal similarity matrices (if compute_similarity=True)
            - 'similarity_analysis': Per-class similarity statistics (if compute_similarity=True)
            - 'output_dir': Path to output directory with all saved files
    
    Example:
        >>> from multiomicsbind import (TemporalMultiOmicsDataset, 
        ...                            train_temporal_model,
        ...                            create_analysis_report)
        >>> 
        >>> # Load data and train model
        >>> dataset = TemporalMultiOmicsDataset(...)
        >>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
        >>> model, history = train_temporal_model(dataset, device, epochs=20)
        >>> 
        >>> # Generate comprehensive report (one line!)
        >>> report = create_analysis_report(
        ...     model, dataset, device, 
        ...     history=history,
        ...     output_dir='./my_analysis',
        ...     class_names=['Responder', 'Non-responder']
        ... )
        >>> 
        >>> # Access results
        >>> print(f"Accuracy: {(report['predictions'] == report['labels']).mean():.4f}")
        >>> print(f"Top 5 features:\\n{report['importance_df'].head()}")
        >>> print(f"\\nAll results saved to: {report['output_dir']}")
    
    Note:
        This function creates the output directory if it doesn't exist and saves:
        - training_history.png (if history provided)
        - confusion_matrix.png
        - classification_report.txt
        - embeddings_umap_{modality}.png for each modality
        - cross_modal_similarity.png (if compute_similarity=True)
        - feature_importance.png (if compute_importance=True)
        - feature_importance_rankings.csv (if compute_importance=True)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 80)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 80)
        print(f"Output directory: {output_path.absolute()}\n")
    
    report_dict = {'output_dir': str(output_path.absolute())}
    
    # 1. Plot training history (if provided)
    if history is not None:
        if verbose:
            print("[1/6] Plotting training history...")
        try:
            history_path = output_path / 'training_history.png'
            plot_training_history_detailed(history, save_path=str(history_path))
            report_dict['training_history_plot'] = str(history_path)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Warning: Could not plot training history: {e}")
    
    # 2. Evaluate model and get embeddings
    if verbose:
        print("[2/6] Evaluating model and extracting embeddings...")
    embeddings, labels, predictions = evaluate_temporal_model(model, dataset, device)
    
    report_dict['embeddings'] = embeddings
    report_dict['labels'] = labels
    report_dict['predictions'] = predictions
    
    # Calculate accuracy
    accuracy = (predictions == labels).mean()
    if verbose:
        print(f"  Model Accuracy: {accuracy:.4f}")
    report_dict['accuracy'] = float(accuracy)
    
    # 3. Generate classification report and confusion matrix
    if verbose:
        print("[3/6] Generating classification metrics...")
    
    # Classification report
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(labels)))]
    
    cls_report = classification_report(labels, predictions, target_names=class_names)
    report_dict['classification_report'] = cls_report
    
    # Save classification report
    report_path = output_path / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(cls_report)
    if verbose:
        print(f"  ✓ Classification report saved to {report_path}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    report_dict['confusion_matrix'] = conf_matrix
    
    # Plot confusion matrix
    cm_path = output_path / 'confusion_matrix.png'
    plot_confusion_matrix(labels, predictions, class_names=class_names, save_path=str(cm_path))
    if verbose:
        print(f"  ✓ Confusion matrix saved to {cm_path}")
    
    # 4. Visualize embeddings with UMAP
    if verbose:
        print("[4/6] Generating UMAP visualizations...")
    
    omics_modalities = [mod for mod in embeddings.keys() 
                       if mod not in ['metadata', 'combined']]
    
    for modality in omics_modalities:
        emb = embeddings[modality]
        
        # Check for NaN values
        if np.isnan(emb).any():
            if verbose:
                print(f"  ⚠ Skipping {modality} (contains NaN values)")
            continue
        
        try:
            umap_path = output_path / f'embeddings_umap_{modality}.png'
            plot_embeddings_umap(emb, labels, title=f'{modality.capitalize()} Embeddings',
                               class_names=class_names,  # ✅ Pass class names
                               save_path=str(umap_path))
            if verbose:
                print(f"  ✓ UMAP plot saved for {modality}")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Could not create UMAP for {modality}: {e}")
    
    # 5. Compute feature importance
    if compute_importance:
        if verbose:
            print(f"[5/6] Computing feature importance ({n_importance_batches} batches)...")
        try:
            importance_dict, importance_df = compute_feature_importance(
                model, dataset, device, 
                n_batches=n_importance_batches,
                verbose=False
            )
            
            report_dict['importance_dict'] = importance_dict
            report_dict['importance_df'] = importance_df
            
            # Save importance rankings
            importance_path = output_path / 'feature_importance_rankings.csv'
            importance_df.to_csv(importance_path, index=False)
            if verbose:
                print(f"  ✓ Feature importance saved to {importance_path}")
            
            # Plot importance distribution
            plot_path = output_path / 'feature_importance.png'
            plot_feature_importance_distribution(
                importance_df, top_k=top_k_features, save_path=str(plot_path)
            )
            if verbose:
                print(f"  ✓ Feature importance plot saved to {plot_path}")
                print(f"\n  Top 5 features overall:")
                for idx, row in importance_df.head(5).iterrows():
                    print(f"    {idx+1}. {row['feature_name']} ({row['modality']}): {row['importance']:.4f}")
        
        except Exception as e:
            if verbose:
                print(f"  ⚠ Warning: Could not compute feature importance: {e}")
    else:
        if verbose:
            print("[5/6] Skipping feature importance computation (disabled)")
    
    # 6. Compute cross-modal similarity
    if compute_similarity:
        if verbose:
            print("[6/6] Computing cross-modal similarity...")
        try:
            similarity_matrices = compute_cross_modal_similarity(embeddings, verbose=False)
            report_dict['similarity_matrices'] = similarity_matrices
            
            # Plot similarity matrices
            sim_path = output_path / 'cross_modal_similarity.png'
            plot_cross_modal_similarity_matrices(similarity_matrices, save_path=str(sim_path))
            if verbose:
                print(f"  ✓ Cross-modal similarity plot saved to {sim_path}")
            
            # Analyze by class
            similarity_analysis = analyze_similarity_by_class(
                similarity_matrices, labels, verbose=False
            )
            report_dict['similarity_analysis'] = similarity_analysis
            
            if verbose:
                print(f"\n  Cross-modal similarity statistics:")
                for comparison, sim_matrix in similarity_matrices.items():
                    mean_sim = np.mean(sim_matrix)
                    print(f"    {comparison}: {mean_sim:.4f}")
        
        except Exception as e:
            if verbose:
                print(f"  ⚠ Warning: Could not compute cross-modal similarity: {e}")
    else:
        if verbose:
            print("[6/6] Skipping cross-modal similarity computation (disabled)")
    
    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Accuracy: {report_dict.get('accuracy', 'N/A'):.4f}")
        print(f"  Samples analyzed: {len(labels)}")
        print(f"  Modalities: {list(embeddings.keys())}")
        print(f"  Output directory: {output_path.absolute()}")
        print("\n" + "=" * 80)
    
    return report_dict
