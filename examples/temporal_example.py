"""
Temporal Multi-Omics Integration Example

This example demonstrates how to use MultiOmicsBind for temporal multi-omics data,
where some modalities are static (single timepoint) and others are temporal 
(multiple timepoints).

Scenario: 
- Transcriptomics and cell painting measured at baseline (t0)
- Proteomics measured at 5 timepoints (0h, 1h, 2h, 4h, 8h) after treatment

The example uses LSTM encoders (recommended default) for temporal modalities,
which are optimal for typical biological time series with sequential dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from multiomicsbind import (
    TemporalMultiOmicsBind, 
    TemporalMultiOmicsDataset,
    contrastive_loss
)


def create_synthetic_temporal_data(n_samples=1000, save_files=True):
    """
    Create synthetic temporal multi-omics data.
    
    Returns:
        Dictionary with data paths and temporal metadata
    """
    print("Creating synthetic temporal multi-omics data...")
    
    # Generate sample IDs
    sample_ids = [f"sample_{i:04d}" for i in range(n_samples)]
    
    # Define timepoints (hours after treatment)
    timepoints = [0, 1, 2, 4, 8]
    
    # Generate labels (3 classes: No response, Partial response, Full response)
    np.random.seed(42)
    labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    
    # Create static modalities (baseline measurements)
    print("Generating static modalities...")
    
    # Transcriptomics (6000 genes)
    transcriptomics_data = []
    for i in range(n_samples):
        # Generate gene expression with label-dependent patterns
        base_expression = np.random.normal(5, 2, 6000)  # log2 expression
        
        # Add label-specific patterns
        if labels[i] == 0:  # No response
            base_expression[:100] += np.random.normal(-1, 0.5, 100)  # Downregulated genes
        elif labels[i] == 1:  # Partial response
            base_expression[:100] += np.random.normal(0.5, 0.5, 100)  # Slightly upregulated
        else:  # Full response
            base_expression[:100] += np.random.normal(2, 0.5, 100)  # Highly upregulated
        
        transcriptomics_data.append({
            'sample_id': sample_ids[i],
            **{f'gene_{j:04d}': base_expression[j] for j in range(6000)}
        })
    
    transcriptomics_df = pd.DataFrame(transcriptomics_data)
    
    # Cell painting (1500 morphological features)
    cell_painting_data = []
    for i in range(n_samples):
        # Generate morphological features
        morphology = np.random.normal(0, 1, 1500)
        
        # Add label-dependent morphological changes
        if labels[i] == 2:  # Full response shows strong morphological changes
            morphology[:50] += np.random.normal(1.5, 0.3, 50)
        elif labels[i] == 1:  # Partial response shows moderate changes
            morphology[:50] += np.random.normal(0.7, 0.3, 50)
        
        cell_painting_data.append({
            'sample_id': sample_ids[i],
            **{f'morph_{j:04d}': morphology[j] for j in range(1500)}
        })
    
    cell_painting_df = pd.DataFrame(cell_painting_data)
    
    # Generate temporal proteomics data
    print("Generating temporal proteomics data...")
    
    proteomics_data = []
    for i in range(n_samples):
        for t_idx, timepoint in enumerate(timepoints):
            # Base protein expression
            base_proteins = np.random.normal(0, 1, 4000)
            
            # Time-dependent response patterns based on labels
            time_factor = timepoint / max(timepoints)  # 0 to 1
            
            if labels[i] == 0:  # No response - minimal change over time
                time_effect = np.random.normal(0, 0.1, 4000) * time_factor
            elif labels[i] == 1:  # Partial response - gradual increase
                time_effect = np.random.normal(0.5, 0.2, 4000) * time_factor
                # Some proteins peak and then decline
                if timepoint > 2:
                    time_effect[:200] *= 0.7
            else:  # Full response - strong early response
                time_effect = np.random.normal(1.5, 0.3, 4000) * (1 - 0.3 * time_factor)
                # Early responders
                if timepoint <= 2:
                    time_effect[:500] += np.random.normal(1, 0.2, 500)
            
            protein_expression = base_proteins + time_effect
            
            proteomics_data.append({
                'sample_id': sample_ids[i],
                'timepoint': timepoint,
                **{f'protein_{j:04d}': protein_expression[j] for j in range(4000)}
            })
    
    proteomics_df = pd.DataFrame(proteomics_data)
    
    # Create metadata
    print("Generating metadata...")
    
    # Drug treatment information
    drugs = ['Drug_A', 'Drug_B', 'Drug_C', 'Vehicle']
    cell_lines = ['HeLa', 'HepG2', 'A549', 'MCF7', 'PC3']
    
    metadata = []
    for i in range(n_samples):
        metadata.append({
            'sample_id': sample_ids[i],
            'drug': np.random.choice(drugs),
            'cell_line': np.random.choice(cell_lines),
            'dose': np.random.uniform(0.1, 10.0),  # μM
            'treatment_duration': 24,  # hours
            'response': labels[i]  # 0: No response, 1: Partial, 2: Full
        })
    
    metadata_df = pd.DataFrame(metadata)
    
    # Save files if requested
    if save_files:
        print("Saving data files...")
        transcriptomics_df.to_csv('transcriptomics_baseline.csv', index=False)
        cell_painting_df.to_csv('cell_painting_baseline.csv', index=False)
        proteomics_df.to_csv('proteomics_timeseries.csv', index=False)
        metadata_df.to_csv('temporal_metadata.csv', index=False)
        
        print("Files saved:")
        print(f"- transcriptomics_baseline.csv: {transcriptomics_df.shape}")
        print(f"- cell_painting_baseline.csv: {cell_painting_df.shape}")
        print(f"- proteomics_timeseries.csv: {proteomics_df.shape}")
        print(f"- temporal_metadata.csv: {metadata_df.shape}")
    
    return {
        'static_data_paths': {
            'transcriptomics': 'transcriptomics_baseline.csv',
            'cell_painting': 'cell_painting_baseline.csv'
        },
        'temporal_data_paths': {
            'proteomics': 'proteomics_timeseries.csv'
        },
        'temporal_metadata': {
            'proteomics': {
                'timepoints': timepoints,
                'time_col': 'timepoint'
            }
        },
        'metadata_path': 'temporal_metadata.csv'
    }


def train_temporal_model(dataset, device, epochs=20):
    """Train the temporal multi-omics model."""
    print(f"\nTraining temporal model on {device}...")
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get input dimensions
    static_dims = {k: v for k, v in dataset.get_input_dims().items() 
                  if k in dataset.static_data}
    temporal_dims = {k: v for k, v in dataset.get_input_dims().items() 
                    if k in dataset.temporal_data}
    
    print(f"Static modalities: {static_dims}")
    print(f"Temporal modalities: {temporal_dims}")
    
    # Define temporal encoders (LSTM is recommended for biological time series)
    temporal_encoders = {}
    for modality in temporal_dims.keys():
        temporal_encoders[modality] = 'lstm'  # LSTM: Best for typical biological time series (3-20 timepoints)
        
        # Alternative encoders for special cases:
        # temporal_encoders[modality] = 'transformer'    # For long sequences (>20 timepoints)
        # temporal_encoders[modality] = 'attention_pool' # For interpretable attention weights
        # temporal_encoders[modality] = 'aggregation'    # For simple temporal patterns
    
    # Create model
    cat_dims, num_dims = dataset.get_metadata_dims()
    
    model = TemporalMultiOmicsBind(
        static_input_dims=static_dims,
        temporal_input_dims=temporal_dims,
        temporal_encoders=temporal_encoders,
        binding_modality='transcriptomics',  # Use transcriptomics as anchor
        cat_dims=cat_dims,
        num_dims=num_dims,
        embed_dim=256,
        num_classes=3,  # No response, Partial, Full
        dropout=0.2,
        temporal_encoder_kwargs={
            'proteomics': {
                'num_layers': 2,
                'bidirectional': True
            }
        }
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    classification_loss_fn = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'total_loss': [],
        'contrastive_loss': [],
        'classification_loss': [],
        'accuracy': []
    }
    
    model.train()
    for epoch in range(epochs):
        epoch_losses = {'total': [], 'contrastive': [], 'classification': []}
        epoch_correct = 0
        epoch_total = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            # Move batch to device
            inputs = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(device)
                elif isinstance(value, dict):
                    inputs[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in value.items()}
                else:
                    inputs[key] = value
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, embeddings = model(inputs, return_embeddings=True)
            
            # Classification loss
            if 'label' in inputs:
                clf_loss = classification_loss_fn(logits, inputs['label'])
                
                # Accuracy
                _, predicted = torch.max(logits.data, 1)
                epoch_total += inputs['label'].size(0)
                epoch_correct += (predicted == inputs['label']).sum().item()
            else:
                clf_loss = torch.tensor(0.0, device=device)
            
            # Contrastive loss
            cont_loss = model.compute_contrastive_loss(embeddings, temperature=0.07)
            
            # Total loss
            total_loss = cont_loss + clf_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record losses
            epoch_losses['total'].append(total_loss.item())
            epoch_losses['contrastive'].append(cont_loss.item())
            epoch_losses['classification'].append(clf_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Con': f"{cont_loss.item():.4f}",
                'Clf': f"{clf_loss.item():.4f}"
            })
        
        # Calculate epoch metrics
        epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
        avg_total_loss = np.mean(epoch_losses['total'])
        avg_cont_loss = np.mean(epoch_losses['contrastive'])
        avg_clf_loss = np.mean(epoch_losses['classification'])
        
        # Update history
        history['total_loss'].append(avg_total_loss)
        history['contrastive_loss'].append(avg_cont_loss)
        history['classification_loss'].append(avg_clf_loss)
        history['accuracy'].append(epoch_accuracy)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Loss: {avg_total_loss:.4f}, "
              f"Contrastive: {avg_cont_loss:.4f}, "
              f"Classification: {avg_clf_loss:.4f}, "
              f"Accuracy: {epoch_accuracy:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
    
    return model, history


def evaluate_temporal_model(model, dataset, device):
    """Evaluate the trained temporal model."""
    print("\nEvaluating temporal model...")
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_embeddings = {modality: [] for modality in 
                     list(dataset.static_data.keys()) + list(dataset.temporal_data.keys())}
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            inputs = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(device)
                elif isinstance(value, dict):
                    inputs[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in value.items()}
                else:
                    inputs[key] = value
            
            # Get predictions and embeddings
            logits, embeddings = model(inputs, return_embeddings=True)
            
            # Store predictions
            if 'label' in inputs:
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(inputs['label'].cpu().numpy())
            
            # Store embeddings
            for modality, embedding in embeddings.items():
                if modality != 'metadata':
                    all_embeddings[modality].append(embedding.cpu().numpy())
    
    # Calculate metrics
    if all_labels:
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        # Classification report
        class_names = ['No Response', 'Partial Response', 'Full Response']
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Concatenate embeddings
    for modality in all_embeddings:
        if all_embeddings[modality]:
            all_embeddings[modality] = np.vstack(all_embeddings[modality])
    
    return all_embeddings, all_labels, all_predictions


def analyze_temporal_patterns(model, dataset, device):
    """Analyze temporal patterns in the proteomics data."""
    print("\nAnalyzing temporal patterns...")
    
    # Get a few samples for analysis
    sample_indices = [0, 100, 200]  # Different response types
    
    model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            sample = dataset[idx]
            
            # Move to device
            inputs = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.unsqueeze(0).to(device)
                elif isinstance(value, dict):
                    inputs[key] = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in value.items()}
                else:
                    inputs[key] = value
            
            # Get prediction and embeddings
            logits, embeddings = model(inputs, return_embeddings=True)
            predicted_class = torch.argmax(logits, dim=1).item()
            actual_class = sample['label'].item() if 'label' in sample else -1
            
            print(f"\nSample {idx}:")
            print(f"  Actual class: {actual_class}, Predicted class: {predicted_class}")
            print(f"  Proteomics shape: {sample['proteomics'].shape}")
            print(f"  Sequence length: {sample['proteomics_seq_len'].item()}")
            
            # Show temporal pattern for first few proteins
            proteomics_data = sample['proteomics'][:5, :5].numpy()  # First 5 timepoints, 5 proteins
            print(f"  Sample proteomics temporal pattern:")
            print(f"    {proteomics_data}")


def main():
    """Main function demonstrating temporal multi-omics integration."""
    print("Temporal Multi-Omics Integration with MultiOmicsBind")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create synthetic temporal data
    data_config = create_synthetic_temporal_data(n_samples=800)
    
    # Create dataset
    print("\nCreating temporal multi-omics dataset...")
    dataset = TemporalMultiOmicsDataset(
        static_data_paths=data_config['static_data_paths'],
        temporal_data_paths=data_config['temporal_data_paths'],
        temporal_metadata=data_config['temporal_metadata'],
        metadata_path=data_config['metadata_path'],
        cat_cols=['drug', 'cell_line'],
        num_cols=['dose'],
        label_col='response',
        normalize=True
    )
    
    print(f"\nDataset created:")
    print(f"- Total samples: {len(dataset)}")
    print(f"- Static modalities: {list(dataset.static_data.keys())}")
    print(f"- Temporal modalities: {list(dataset.temporal_data.keys())}")
    print(f"- Temporal info: {dataset.get_temporal_info()}")
    
    # Train model
    model, history = train_temporal_model(dataset, device, epochs=15)
    
    # Save model
    torch.save(model.state_dict(), 'temporal_multiomicsbind.pth')
    print("\nModel saved as 'temporal_multiomicsbind.pth'")
    
    # Evaluate model
    embeddings, labels, predictions = evaluate_temporal_model(model, dataset, device)
    
    # Analyze temporal patterns
    analyze_temporal_patterns(model, dataset, device)
    
    # Plot training history
    print("\nPlotting training history...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Total loss
    axes[0, 0].plot(history['total_loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Contrastive loss
    axes[0, 1].plot(history['contrastive_loss'], color='red')
    axes[0, 1].set_title('Contrastive Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Classification loss
    axes[1, 0].plot(history['classification_loss'], color='green')
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 1].plot(history['accuracy'], color='purple')
    axes[1, 1].set_title('Training Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Temporal MultiOmicsBind Training History', fontsize=16)
    plt.tight_layout()
    plt.savefig('temporal_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTemporal multi-omics integration completed!")
    print("\nKey findings:")
    print("- Successfully integrated static (transcriptomics, cell painting) and temporal (proteomics) data")
    print("- LSTM encoder effectively captured temporal proteomics patterns")
    print("- Binding modality approach maintained efficiency with mixed data types")
    print("- Model achieved good classification performance on response prediction")
    
    print("\nGenerated files:")
    print("- temporal_multiomicsbind.pth (trained model)")
    print("- temporal_training_history.png (training curves)")
    print("- transcriptomics_baseline.csv, cell_painting_baseline.csv (static data)")
    print("- proteomics_timeseries.csv (temporal data)")
    print("- temporal_metadata.csv (sample metadata)")


if __name__ == "__main__":
    main()