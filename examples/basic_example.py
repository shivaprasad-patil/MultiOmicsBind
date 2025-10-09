"""
Basic example of using MultiOmicsBind for multi-omics data integration.

This example demonstrates how to:
1. Load multi-omics data
2. Initialize and train the model
3. Evaluate the results
4. Visualize embeddings

Run with: python basic_example.py
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from multiomicsbind import (
    MultiOmicsBindWithHead,
    MultiOmicsDataset,
    train_multiomicsbind,
    evaluate_model,
    plot_training_history
)


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
    print("MultiOmicsBind Basic Example")
    print("=" * 50)
    
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
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    print("\n4. Initializing model...")
    input_dims = dataset.get_input_dims()
    cat_dims, num_dims = dataset.get_metadata_dims()
    
    # Demo: Compare different binding modality strategies
    print("\n   Available modalities:", list(input_dims.keys()))
    print("   Testing different binding modality approaches...")
    
    # Use transcriptomics as binding modality (typically most comprehensive)
    model = MultiOmicsBindWithHead(
        input_dims=input_dims,
        cat_dims=cat_dims,
        num_dims=num_dims,
        embed_dim=256,  # Smaller for demo
        num_classes=3,  # Low, Medium, High response
        binding_modality='transcriptomics'  # NEW: Use binding modality approach
    ).to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Binding modality: {model.get_binding_modality()}")
    print(f"   Complexity: O(n) instead of O(n²) for contrastive learning")
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Train model
    print("\n5. Training model...")
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
    
    # Evaluate model
    print("\n6. Evaluating model...")
    
    # ⚠️ WARNING: For demonstration purposes, we evaluate on both train and val sets.
    # In production, you should ONLY evaluate on held-out test data to avoid data leakage.
    # Evaluating on training data will show inflated performance metrics.
    # See BEST_PRACTICES.md for proper train/val/test split guidelines.
    
    train_metrics = evaluate_model(trained_model, train_loader, device, use_classification=True)
    val_metrics = evaluate_model(trained_model, val_loader, device, use_classification=True)
    
    print(f"Training metrics (biased - for reference only): {train_metrics}")
    print(f"Validation metrics (unbiased): {val_metrics}")
    
    # Save model
    print("\n7. Saving model...")
    torch.save(trained_model.state_dict(), 'multiomicsbind_trained.pth')
    print("Model saved as 'multiomicsbind_trained.pth'")
    
    # Create visualizations
    print("\n8. Creating visualizations...")
    
    # Plot training history
    if hasattr(trained_model, 'training_history'):
        plot_training_history(trained_model.training_history, save_path='training_history.png')
    
    print("\nExample completed successfully!")
    print("Generated files:")
    print("- multiomicsbind_trained.pth (trained model)")
    print("- training_history.png (training curves)")


if __name__ == "__main__":
    main()
