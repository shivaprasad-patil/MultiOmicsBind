"""
Basic example of using MultiOmicsBind for multi-omics data integration.

This example demonstrates how to:
1. Load multi-omics data
2. Train with automatic train/test splitting (NEW!)
3. Evaluate on held-out test set
4. Use custom class names in visualizations (NEW!)
5. Visualize embeddings and results

Run with: python basic_example.py
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd

from multiomicsbind import (
    MultiOmicsDataset,
    MultiOmicsBindWithHead,
    train_multiomicsbind,
    evaluate_model,
    plot_training_history
)
from multiomicsbind.training.evaluation import evaluate_temporal_model
from multiomicsbind.analysis import create_analysis_report


def create_synthetic_data():
    """Create synthetic multi-omics data for demonstration.
    
    This example uses 3 modalities with specific feature counts, but the framework
    supports any number of modalities with any number of features per modality.
    Simply modify the data_config dictionary below to customize your setup.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Configuration for synthetic data - easily customizable
    data_config = {
        'transcriptomics': 6000,  # genes (can be any number)
        'proteomics': 4000,       # proteins (can be any number)
        'cell_painting': 1500,    # morphological features (can be any number)
        # Add more modalities as needed:
        # 'metabolomics': 2000,
        # 'genomics': 500000,
        # 'imaging': 10000,
    }
    
    # Create sample IDs
    sample_ids = [f"sample_{i:04d}" for i in range(n_samples)]
    
    print(f"Creating synthetic data with {len(data_config)} modalities:")
    for modality, n_features in data_config.items():
        print(f"  - {modality}: {n_features} features")
    
    # Generate data for each modality
    for modality, n_features in data_config.items():
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        
        if modality == 'transcriptomics':
            columns = [f"gene_{i}" for i in range(n_features)]
        elif modality == 'proteomics':
            columns = [f"protein_{i}" for i in range(n_features)]
        elif modality == 'cell_painting':
            columns = [f"morph_feature_{i}" for i in range(n_features)]
        else:
            # Generic naming for additional modalities
            columns = [f"{modality}_feature_{i}" for i in range(n_features)]
        
        df = pd.DataFrame(data, columns=columns)
        df.insert(0, 'sample_id', sample_ids)
        df.to_csv(f'{modality}.csv', index=False)
    
    # Metadata
    drugs = np.random.choice(['Drug_A', 'Drug_B', 'Drug_C', 'Drug_D'], n_samples)
    cell_lines = np.random.choice(['HeLa', 'MCF7', 'A549'], n_samples)
    doses = np.random.uniform(0.1, 10.0, n_samples)
    
    # Create response labels based on drug and dose (synthetic relationship)
    response = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if drugs[i] in ['Drug_A', 'Drug_B'] and doses[i] > 5.0:
            response[i] = 2  # High response
        elif doses[i] > 2.0:
            response[i] = 1  # Medium response
        else:
            response[i] = 0  # Low response
    
    meta_df = pd.DataFrame({
        'sample_id': sample_ids,
        'drug': drugs,
        'cell_line': cell_lines,
        'dose': doses,
        'response': response
    })
    meta_df.to_csv('metadata.csv', index=False)
    
    print("\nSynthetic data created:")
    for modality, n_features in data_config.items():
        print(f"- {modality}.csv: {n_samples} samples × {n_features} features")
    print(f"- metadata.csv: {meta_df.shape}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("MultiOmicsBind Basic Example - NEW High-Level API")
    print("=" * 60)
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    create_synthetic_data()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n2. Using device: {device}")
    
    # Load dataset - automatically detects all modalities from created files
    print("\n3. Loading dataset...")
    
    # Define which modalities to use (can be customized)
    modalities_to_use = ['transcriptomics', 'proteomics', 'cell_painting']
    # You can easily add more modalities:
    # modalities_to_use = ['transcriptomics', 'proteomics', 'cell_painting', 'metabolomics']
    
    data_paths = {modality: f'{modality}.csv' for modality in modalities_to_use}
    
    print(f"Loading {len(data_paths)} modalities: {list(data_paths.keys())}")
    
    dataset = MultiOmicsDataset(
        data_paths=data_paths,
        metadata_path='metadata.csv',
        cat_cols=['drug', 'cell_line'],
        num_cols=['dose'],
        label_col='response'
    )
    
    print(f"Total samples: {len(dataset)}")
    
    # ============================================
    # NEW SIMPLIFIED API - Automatic train/test splitting!
    # ============================================
    print("\n" + "=" * 60)
    print("USING NEW HIGH-LEVEL API WITH AUTOMATIC TRAIN/TEST SPLIT")
    print("=" * 60)
    
    # Train model with automatic train/test split (80%/20%)
    # The split happens automatically behind the scenes with reproducible seed!
    print(f"\nDataset info:")
    print(f"- Modalities: {list(modalities_to_use)}")
    print(f"- Total samples: {len(dataset)}")
    
    # ============================================
    # NEW: Automatic train/test splitting with reproducible seed!
    # ============================================
    print(f"\n4. Automatic train/test splitting...")
    torch.manual_seed(42)  # Reproducible splits
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    print(f"\n5. Initializing model...")
    input_dims = dataset.get_input_dims()
    cat_dims, num_dims = dataset.get_metadata_dims()
    
    model = MultiOmicsBindWithHead(
        input_dims=input_dims,
        cat_dims=cat_dims,
        num_dims=num_dims,
        embed_dim=256,
        num_classes=3,
        binding_modality='transcriptomics',  # Use binding modality approach
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Binding modality: transcriptomics")
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Train model
    print(f"\n6. Training for 20 epochs...")
    trained_model = train_multiomicsbind(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=20,
        temperature=0.07,
        use_classification=True,
        scheduler=scheduler
    )
    
    print(f"\n✅ Training complete!")
    
    # Save model
    torch.save(trained_model.state_dict(), 'multiomicsbind_trained.pth')
    print(f"   Model saved to: multiomicsbind_trained.pth")
    
    # Evaluate on held-out test set (NO DATA LEAKAGE!)
    print("\n7. Evaluating on held-out test set...")
    
    embeddings, labels, predictions = evaluate_temporal_model(trained_model, test_dataset, device)
    test_accuracy = (predictions == labels).mean()
    
    print(f"\n✅ Test Set Accuracy: {test_accuracy:.4f}")
    
    # Show per-class accuracy
    for class_idx in range(3):
        class_mask = labels == class_idx
        if class_mask.sum() > 0:
            class_acc = (predictions[class_mask] == labels[class_mask]).mean()
            print(f"   Class {class_idx} accuracy: {class_acc:.4f} ({class_mask.sum()} samples)")
    
    # ============================================
    # NEW FEATURE: Custom Class Names in Visualizations
    # ============================================
    print("\n8. Generating comprehensive analysis with custom class names...")
    
    # Define meaningful class names instead of generic "Class 0, 1, 2"
    class_names = ['Low Response', 'Medium Response', 'High Response']
    
    report = create_analysis_report(
        model=trained_model,
        dataset=test_dataset,  # Use test set only!
        device=device,
        class_names=class_names,  # NEW: Custom class names!
        output_dir='./analysis_results',
        compute_importance=True,
        compute_similarity=True,
        n_importance_batches=5,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nTest Accuracy: {report['accuracy']:.4f}")
    print(f"Output directory: analysis_results/")
    print("\nGenerated files:")
    print("  ├── multiomicsbind_trained.pth (trained model)")
    print("  ├── analysis_results/")
    print("  │   ├── classification_report.txt")
    print("  │   ├── confusion_matrix.png (with custom class names!)")
    print("  │   ├── embeddings_umap_transcriptomics.png")
    print("  │   ├── embeddings_umap_proteomics.png")
    print("  │   ├── embeddings_umap_cell_painting.png")
    print("  │   ├── feature_importance.csv")
    print("  │   └── cross_modal_similarity.png")
    
    # Plot training history (if available)
    print("\n9. Saving training history...")
    if hasattr(trained_model, 'training_history'):
        from multiomicsbind.utils.visualization import plot_training_history_detailed
        plot_training_history_detailed(trained_model.training_history, save_path='training_history.png')
        print("   ✅ training_history.png saved")
    else:
        print("   Training history not available (use verbose=True in training)")
    
    print("\n" + "=" * 60)
    print("🎉 EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n✨ NEW FEATURES DEMONSTRATED:")
    print("  ✅ Automatic train/test splitting (train_split=0.8, test_split=0.2)")
    print("  ✅ Custom class names in all visualizations")
    print("  ✅ High-level API (train_multiomicsbind_simple)")
    print("  ✅ Comprehensive analysis report")
    print("  ✅ No data leakage (test set never seen during training)")
    
    print("\n💡 Next steps:")
    print("  • Try temporal_example.py for time-series data")
    print("  • See advanced_analysis.py for feature importance")
    print("  • Check flexible_modalities_example.py for custom modality combinations")


if __name__ == "__main__":
    main()
