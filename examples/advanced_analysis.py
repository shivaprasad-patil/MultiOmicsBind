"""
Advanced example showing feature interpretation and embedding analysis.

This example demonstrates:
1. Model interpretation using gradient-based attribution
2. Embedding space analysis with UMAP
3. Feature importance ranking
4. Cross-modal similarity analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from multiomicsbind import (
    MultiOmicsBindWithHead,
    MultiOmicsDataset,
    plot_embeddings_umap,
    plot_feature_importance,
    plot_confusion_matrix
)


def get_gradients(model, inputs, target_class=None):
    """
    Get gradients for feature attribution analysis.
    
    Args:
        model: Trained MultiOmicsBind model
        inputs: Input data batch
        target_class: Target class for gradient computation
        
    Returns:
        Dict of gradients for each modality
    """
    model.eval()
    
    # Enable gradients for inputs
    for modality, data in inputs.items():
        if isinstance(data, torch.Tensor) and modality != 'label':
            data.requires_grad_(True)
    
    # Forward pass
    logits, embeddings = model(inputs, return_embeddings=True)
    
    if target_class is None:
        # Use predicted class
        target_class = torch.argmax(logits, dim=1)
    
    # Compute gradients
    gradients = {}
    for modality, data in inputs.items():
        if isinstance(data, torch.Tensor) and modality != 'label' and modality != 'metadata':
            # Get gradient of logits w.r.t. input
            grad = torch.autograd.grad(
                outputs=logits[0, target_class[0]], 
                inputs=data,
                retain_graph=True
            )[0]
            gradients[modality] = grad.detach().cpu().numpy()
    
    return gradients


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


def compute_cross_modal_similarity(embeddings):
    """
    Compute cross-modal similarity matrices.
    
    Args:
        embeddings: Dictionary of embeddings for each modality
        
    Returns:
        Dictionary of similarity matrices
    """
    similarities = {}
    modalities = list(embeddings.keys())
    
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i < j:  # Avoid duplicate pairs
                # Normalize embeddings
                emb1_norm = F.normalize(torch.tensor(embeddings[mod1]), p=2, dim=1)
                emb2_norm = F.normalize(torch.tensor(embeddings[mod2]), p=2, dim=1)
                
                # Compute cosine similarity
                similarity = torch.mm(emb1_norm, emb2_norm.T).numpy()
                similarities[f"{mod1}_vs_{mod2}"] = similarity
    
    return similarities


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
        num_classes=3
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
        plot_embeddings_umap(
            embeddings, labels, 
            save_path='embeddings_umap.png',
            figsize=(15, 5)
        )
        print("✓ UMAP visualization saved as 'embeddings_umap.png'")
    except ImportError:
        print("✗ UMAP not available. Install with: pip install umap-learn")
    
    # 4. Feature importance analysis
    print("\n4. Analyzing feature importance...")
    
    # Get a sample batch for gradient analysis
    sample_batch = next(iter(dataloader))
    sample_inputs = {}
    for k, v in sample_batch.items():
        if isinstance(v, dict):
            sample_inputs[k] = {k2: v2[:1].to(device) for k2, v2 in v.items()}
        elif isinstance(v, torch.Tensor):
            sample_inputs[k] = v[:1].to(device)
        else:
            sample_inputs[k] = v
    
    try:
        gradients = get_gradients(model, sample_inputs)
        
        # Plot feature importance for each modality
        for modality, grad in gradients.items():
            feature_names = dataset.get_feature_names(modality)
            if len(feature_names) == grad.shape[1]:
                importance_scores = np.mean(np.abs(grad), axis=0)
                
                plot_feature_importance(
                    importance_scores,
                    feature_names,
                    top_k=20,
                    title=f"{modality.title()} Feature Importance",
                    save_path=f'feature_importance_{modality}.png'
                )
                print(f"✓ Feature importance for {modality} saved")
    
    except Exception as e:
        print(f"✗ Feature importance analysis failed: {e}")
    
    # 5. Cross-modal similarity analysis
    print("\n5. Computing cross-modal similarities...")
    
    similarities = compute_cross_modal_similarity(embeddings)
    
    print("Cross-modal similarity statistics:")
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
    
    print("\nAdvanced analysis completed!")
    print("\nGenerated files:")
    print("- embeddings_umap.png (UMAP visualization)")
    print("- feature_importance_*.png (feature importance plots)")
    print("- confusion_matrix.png (classification results)")


if __name__ == "__main__":
    main()
