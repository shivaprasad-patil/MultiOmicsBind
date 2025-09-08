"""
Advanced example showing MultiOmicsBind's flexibility with different numbers
of modalities and feature dimensions.

This example demonstrates:
1. How to easily customize the number of modalities
2. How to handle different feature dimensions
3. How to scale to large datasets
4. How to add new modality types

Run with: python flexible_modalities_example.py
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
    plot_architecture
)


def create_custom_synthetic_data(data_config, n_samples=1000):
    """
    Create synthetic multi-omics data with custom configuration.
    
    Args:
        data_config (dict): Configuration specifying modalities and their feature counts
        n_samples (int): Number of samples to generate
    """
    np.random.seed(42)
    
    # Create sample IDs
    sample_ids = [f"sample_{i:04d}" for i in range(n_samples)]
    
    print(f"Creating synthetic data with {len(data_config)} modalities:")
    
    # Generate data for each modality
    for modality, n_features in data_config.items():
        print(f"  - {modality}: {n_features:,} features")
        
        # Generate data with some realistic distributions
        if 'genomics' in modality.lower() or 'snp' in modality.lower():
            # Binary/categorical data for genomics
            data = np.random.choice([0, 1, 2], size=(n_samples, n_features), p=[0.7, 0.2, 0.1])
        elif 'imaging' in modality.lower():
            # Positive values for imaging features
            data = np.random.exponential(scale=1.0, size=(n_samples, n_features))
        else:
            # Standard normal distribution for other omics
            data = np.random.randn(n_samples, n_features)
        
        # Create appropriate column names
        if 'transcriptomics' in modality.lower():
            columns = [f"gene_{i}" for i in range(n_features)]
        elif 'proteomics' in modality.lower():
            columns = [f"protein_{i}" for i in range(n_features)]
        elif 'metabolomics' in modality.lower():
            columns = [f"metabolite_{i}" for i in range(n_features)]
        elif 'cell_painting' in modality.lower():
            columns = [f"morph_feature_{i}" for i in range(n_features)]
        elif 'genomics' in modality.lower():
            columns = [f"snp_{i}" for i in range(n_features)]
        elif 'imaging' in modality.lower():
            columns = [f"image_feature_{i}" for i in range(n_features)]
        else:
            columns = [f"{modality}_feature_{i}" for i in range(n_features)]
        
        # Save to CSV
        df = pd.DataFrame(data.astype(np.float32), columns=columns)
        df.insert(0, 'sample_id', sample_ids)
        df.to_csv(f'{modality}.csv', index=False)
    
    # Create metadata
    drugs = np.random.choice(['Drug_A', 'Drug_B', 'Drug_C', 'Drug_D', 'Drug_E'], n_samples)
    cell_lines = np.random.choice(['HeLa', 'MCF7', 'A549', 'HEK293'], n_samples)
    doses = np.random.uniform(0.1, 10.0, n_samples)
    time_points = np.random.choice([6, 12, 24, 48], n_samples)  # hours
    
    # Create complex response based on multiple factors
    response = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        score = 0
        if drugs[i] in ['Drug_A', 'Drug_B']:
            score += 1
        if doses[i] > 5.0:
            score += 1
        if time_points[i] >= 24:
            score += 1
        if cell_lines[i] in ['HeLa', 'MCF7']:
            score += 1
        
        # Convert score to response categories
        if score >= 3:
            response[i] = 2  # High response
        elif score >= 2:
            response[i] = 1  # Medium response
        else:
            response[i] = 0  # Low response
    
    meta_df = pd.DataFrame({
        'sample_id': sample_ids,
        'drug': drugs,
        'cell_line': cell_lines,
        'dose': doses,
        'time_point': time_points,
        'response': response
    })
    meta_df.to_csv('metadata.csv', index=False)
    
    print(f"\nSynthetic data created:")
    for modality, n_features in data_config.items():
        print(f"- {modality}.csv: {n_samples:,} samples × {n_features:,} features")
    print(f"- metadata.csv: {meta_df.shape}")


def example_small_study():
    """Example with minimal modalities for quick testing."""
    print("=" * 60)
    print("EXAMPLE 1: Small Study (3 modalities)")
    print("=" * 60)
    
    data_config = {
        'transcriptomics': 1000,    # 1K genes
        'proteomics': 500,          # 500 proteins
        'metabolomics': 200         # 200 metabolites
    }
    
    return run_experiment(data_config, embed_dim=128, epochs=10)


def example_medium_study():
    """Example with moderate number of modalities and features."""
    print("=" * 60)
    print("EXAMPLE 2: Medium Study (4 modalities)")
    print("=" * 60)
    
    data_config = {
        'transcriptomics': 10000,   # 10K genes
        'proteomics': 5000,         # 5K proteins
        'cell_painting': 1500,      # 1.5K morphological features
        'metabolomics': 3000        # 3K metabolites
    }
    
    return run_experiment(data_config, embed_dim=256, epochs=15)


def example_large_study():
    """Example with many modalities including high-dimensional genomics."""
    print("=" * 60)
    print("EXAMPLE 3: Large Study (6 modalities)")
    print("=" * 60)
    
    data_config = {
        'transcriptomics': 25000,   # 25K genes
        'proteomics': 8000,         # 8K proteins
        'metabolomics': 2500,       # 2.5K metabolites
        'cell_painting': 1500,      # 1.5K morphological features
        'genomics': 50000,          # 50K SNPs
        'imaging': 10000            # 10K imaging features
    }
    
    return run_experiment(data_config, embed_dim=512, epochs=5, batch_size=16)


def run_experiment(data_config, embed_dim=256, epochs=10, batch_size=32):
    """
    Run a multi-omics experiment with given configuration.
    
    Args:
        data_config (dict): Modality configuration
        embed_dim (int): Embedding dimension
        epochs (int): Number of training epochs
        batch_size (int): Batch size
    
    Returns:
        dict: Experiment results
    """
    # Create synthetic data
    create_custom_synthetic_data(data_config, n_samples=500)  # Smaller for demo
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load dataset
    data_paths = {modality: f'{modality}.csv' for modality in data_config.keys()}
    
    dataset = MultiOmicsDataset(
        data_paths=data_paths,
        metadata_path='metadata.csv',
        cat_cols=['drug', 'cell_line'],
        num_cols=['dose', 'time_point'],
        label_col='response'
    )
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataset info:")
    print(f"- Modalities: {list(data_config.keys())}")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    
    # Initialize model
    input_dims = dataset.get_input_dims()
    cat_dims, num_dims = dataset.get_metadata_dims()
    
    print(f"\nModel configuration:")
    print(f"- Input dimensions: {input_dims}")
    print(f"- Embedding dimension: {embed_dim}")
    
    model = MultiOmicsBindWithHead(
        input_dims=input_dims,
        cat_dims=cat_dims,
        num_dims=num_dims,
        embed_dim=embed_dim,
        num_classes=3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"- Total parameters: {total_params:,}")
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    trained_model = train_multiomicsbind(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        temperature=0.07,
        use_classification=True,
        scheduler=scheduler
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    train_metrics = evaluate_model(trained_model, train_loader, device, use_classification=True)
    val_metrics = evaluate_model(trained_model, val_loader, device, use_classification=True)
    
    print(f"Training metrics: {train_metrics}")
    print(f"Validation metrics: {val_metrics}")
    
    # Create architecture visualization
    plot_architecture(
        save_path=f'architecture_{len(data_config)}_modalities.png',
        custom_modalities=data_config
    )
    print(f"Architecture saved as 'architecture_{len(data_config)}_modalities.png'")
    
    return {
        'data_config': data_config,
        'model_params': total_params,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }


def main():
    """Run multiple examples showing flexibility."""
    print("MultiOmicsBind Flexibility Examples")
    print("=" * 80)
    
    results = []
    
    # Run different scale examples
    try:
        results.append(example_small_study())
    except Exception as e:
        print(f"Small study failed: {e}")
    
    try:
        results.append(example_medium_study())
    except Exception as e:
        print(f"Medium study failed: {e}")
    
    try:
        results.append(example_large_study())
    except Exception as e:
        print(f"Large study failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF EXPERIMENTS")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        data_config = result['data_config']
        total_features = sum(data_config.values())
        
        print(f"\nExperiment {i}:")
        print(f"  Modalities: {len(data_config)} ({', '.join(data_config.keys())})")
        print(f"  Total features: {total_features:,}")
        print(f"  Model parameters: {result['model_params']:,}")
        print(f"  Validation accuracy: {result['val_metrics'].get('accuracy', 'N/A'):.3f}")
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS:")
    print("- MultiOmicsBind automatically adapts to any number of modalities")
    print("- Feature dimensions can range from hundreds to hundreds of thousands")
    print("- The architecture scales efficiently with data complexity")
    print("- Easy to add new modality types by simply adding data files")
    print("=" * 80)


if __name__ == "__main__":
    main()
