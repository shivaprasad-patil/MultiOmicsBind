"""
Advanced example showing feature interpretation and embedding analysis.

This example demonstrates:
1. Model interpretation using gradient-based attribution
2. Embedding space analysis with UMAP
3. Feature importance ranking
4. Cross-modal similarity analysis

Updated to use the new high-level API functions.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from multiomicsbind import (
    MultiOmicsBindWithHead,
    MultiOmicsDataset,
    compute_feature_importance,      # NEW: High-level API
    compute_cross_modal_similarity,  # NEW: High-level API
    plot_embeddings_umap,
    plot_feature_importance,
    plot_confusion_matrix,
    set_seed                         # NEW: Reproducibility
)


def analyze_embeddings(model, dataloader, device):
    """
    Extract and analyze embeddings from trained model.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Torch device
        
    Returns:
        Dictionary containing embeddings and labels
    """
    model.eval()
    
    all_embeddings = {modality: [] for modality in ['transcriptomics', 'proteomics', 'cell_painting']}
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            inputs = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    inputs[k] = {k2: v2.to(device) for k2, v2 in v.items()}
                elif isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                else:
                    inputs[k] = v
            
            # Get embeddings
            embeddings = model.encode(inputs)
            
            # Store embeddings
            for modality in all_embeddings.keys():
                if modality in embeddings:
                    all_embeddings[modality].append(embeddings[modality].cpu().numpy())
            
            # Store labels
            if 'label' in inputs:
                all_labels.append(inputs['label'].cpu().numpy())
    
    # Concatenate results
    for modality in all_embeddings.keys():
        if all_embeddings[modality]:
            all_embeddings[modality] = np.vstack(all_embeddings[modality])
        else:
            del all_embeddings[modality]
    
    all_labels = np.concatenate(all_labels) if all_labels else None
    
    return all_embeddings, all_labels


def main():
    """Main analysis function."""
    print("MultiOmicsBind Advanced Analysis")
    print("=" * 50)
    
    # Load pre-trained model (assumes basic_example.py was run first)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n1. Loading dataset and model...")
    
    # Load dataset
    data_paths = {
        'transcriptomics': 'transcriptomics.csv',
        'proteomics': 'proteomics.csv',
        'cell_painting': 'cell_painting.csv'
    }
    
    dataset = MultiOmicsDataset(
        data_paths=data_paths,
        metadata_path='metadata.csv',
        cat_cols=['drug', 'cell_line'],
        num_cols=['dose'],
        label_col='response'
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load trained model
    input_dims = dataset.get_input_dims()
    cat_dims, num_dims = dataset.get_metadata_dims()
    
    model = MultiOmicsBindWithHead(
        input_dims=input_dims,
        cat_dims=cat_dims,
        num_dims=num_dims,
        embed_dim=256,
        num_classes=3,
        binding_modality='transcriptomics'  # Use binding modality for efficiency
    ).to(device)
    
    try:
        model.load_state_dict(torch.load('multiomicsbind_trained.pth', map_location=device))
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("✗ Trained model not found. Please run basic_example.py first.")
        return
    
    # 2. Extract embeddings
    print("\n2. Extracting embeddings...")
    embeddings, labels = analyze_embeddings(model, dataloader, device)
    
    print(f"Extracted embeddings for {len(embeddings)} modalities:")
    for modality, emb in embeddings.items():
        print(f"  - {modality}: {emb.shape}")
    
    # 3. Visualize embeddings with UMAP
    print("\n3. Creating UMAP visualizations...")
    try:
        # Use the first modality's embeddings for UMAP visualization
        if embeddings and labels is not None:
            first_modality = list(embeddings.keys())[0]
            first_embeddings = embeddings[first_modality]
            
            plot_embeddings_umap(
                first_embeddings, labels, 
                save_path='embeddings_umap.png',
                figsize=(10, 8),
                class_names=['Low', 'Medium', 'High']
            )
            print(f"✓ UMAP visualization saved for {first_modality} embeddings")
        else:
            print("✗ No embeddings or labels available for UMAP")
    except ImportError:
        print("✗ UMAP not available. Install with: pip install umap-learn")
    except Exception as e:
        print(f"✗ UMAP visualization failed: {e}")
    
    # 4. Feature importance analysis
    print("\n4. Analyzing feature importance...")
    
    # Use the new high-level API for feature importance
    try:
        importance_dict, importance_df = compute_feature_importance(
            model=model,
            dataset=dataset,
            device=device,
            n_batches=10,
            batch_size=16,
            verbose=True
        )
        
        print(f"✓ Feature importance computed for all modalities")
        print(f"  Total features analyzed: {len(importance_df)}")
        
        # Plot feature importances
        plot_feature_importance(
            importance_dict,
            save_path='feature_importance_all.png',
            top_k=20
        )
        print(f"✓ Feature importance plot saved")
        
        # Save feature importance DataFrame
        importance_df.to_csv('feature_importance.csv', index=False)
        print(f"✓ Feature importance CSV saved")
    
    except Exception as e:
        print(f"✗ Feature importance analysis failed: {e}")
    
    # 5. Cross-modal similarity analysis
    print("\n5. Computing cross-modal similarities...")
    
    # Use the new high-level API for cross-modal similarity
    similarities = compute_cross_modal_similarity(
        embeddings_dict=embeddings,
        verbose=True
    )
    
    print("\nCross-modal similarity statistics:")
    for pair, sim_matrix in similarities.items():
        mean_sim = np.mean(np.diag(sim_matrix))  # Diagonal = same sample similarity
        print(f"  - {pair}: {mean_sim:.3f} (same sample similarity)")
    
    # 6. Classification performance analysis
    print("\n6. Detailed classification analysis...")
    
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    inputs[k] = {k2: v2.to(device) for k2, v2 in v.items()}
                elif isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                else:
                    inputs[k] = v
            
            if 'label' in inputs:
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                true = inputs['label'].cpu().numpy()
                
                all_preds.extend(preds)
                all_true.extend(true)
    
    if all_true and all_preds:
        # Classification report
        label_names = ['Low', 'Medium', 'High']
        print("\nClassification Report:")
        print(classification_report(all_true, all_preds, target_names=label_names))
        
        # Confusion matrix
        plot_confusion_matrix(
            np.array(all_true), 
            np.array(all_preds),
            class_names=label_names,
            save_path='confusion_matrix.png'
        )
        print("✓ Confusion matrix saved as 'confusion_matrix.png'")
    
    print("\n" + "=" * 50)
    print("ADVANCED ANALYSIS COMPLETE!")
    print("=" * 50)
    
    print("\nGenerated files:")
    print("- embeddings_umap.png (UMAP visualization)")
    print("- feature_importance_all.png (feature importance plots)")
    print("- feature_importance.csv (detailed importance scores)")
    print("- confusion_matrix.png (classification results)")
    
    print("\nUsing new high-level API functions:")
    print("✓ compute_feature_importance() - Gradient-based analysis")
    print("✓ compute_cross_modal_similarity() - Cross-modal analysis")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    main()
