#!/usr/bin/env python3
"""
Example demonstrating the Binding Modality concept in MultiOmicsBind.

This example shows how to use one modality as an anchor to align all other modalities,
similar to how ImageBind uses vision as the binding modality.
"""

import torch
import numpy as np
from multiomicsbind import MultiOmicsBindWithHead, MultiOmicsDataset, train_multiomicsbind
from multiomicsbind.core.losses import binding_modality_loss, contrastive_loss
import matplotlib.pyplot as plt


def create_synthetic_multiomics_data(n_samples=500, seed=42):
    """Create synthetic multi-omics data for demonstration."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Define modalities with different dimensions
    modalities = {
        'transcriptomics': 5000,  # Gene expression (largest, most comprehensive)
        'proteomics': 2000,       # Protein levels
        'metabolomics': 800,      # Metabolite concentrations
        'cell_painting': 1200,    # Morphological features
        'genomics': 10000         # SNP data
    }
    
    # Create correlated data (samples from same condition should be similar)
    n_conditions = 3
    condition_labels = np.random.randint(0, n_conditions, n_samples)
    
    data = {}
    
    # Generate shared biological signals for correlation between modalities
    max_features = max(modalities.values())
    shared_signals = []
    for i in range(n_samples):
        condition = condition_labels[i]
        shared_signal = np.random.randn(max_features) * 0.5 + condition * 2
        shared_signals.append(shared_signal)
    
    for modality, n_features in modalities.items():
        modality_data = []
        
        for i in range(n_samples):
            # Base signal for condition
            condition = condition_labels[i]
            base_signal = np.random.randn(n_features) * 0.5 + condition * 2
            
            # Add modality-specific noise
            noise = np.random.randn(n_features) * 0.8
            
            # Add some correlation between modalities (shared biological signal)
            if modality == 'transcriptomics':
                # Transcriptomics as the most comprehensive modality - use shared signal directly
                sample = shared_signals[i][:n_features] + noise
            else:
                # Other modalities share some signal with transcriptomics
                shared_component = shared_signals[i][:n_features] * 0.6  # Partial overlap
                sample = shared_component + base_signal * 0.4 + noise
            
            modality_data.append(sample)
        
        data[modality] = np.array(modality_data, dtype=np.float32)
    
    return data, condition_labels


def demonstrate_binding_modality():
    """Demonstrate the binding modality concept."""
    print("🧬 MultiOmicsBind Binding Modality Demonstration")
    print("=" * 60)
    
    # Create synthetic data
    print("\n📊 Creating synthetic multi-omics data...")
    data, labels = create_synthetic_multiomics_data(n_samples=200)
    
    modalities = list(data.keys())
    print(f"Modalities: {modalities}")
    for mod, arr in data.items():
        print(f"  {mod}: {arr.shape}")
    
    # Convert to tensors
    batch_size = 32
    embeddings = {}
    
    # Simulate embeddings (normally these would come from encoders)
    embed_dim = 512
    for modality in modalities:
        # Normalize the data and project to embedding dimension
        normalized_data = torch.tensor(data[modality][:batch_size])
        # Simple linear projection for demonstration
        projection = torch.randn(data[modality].shape[1], embed_dim) * 0.1
        embeddings[modality] = torch.mm(normalized_data, projection)
        embeddings[modality] = torch.nn.functional.normalize(embeddings[modality], dim=1)
    
    print(f"\n🎯 Testing different binding modality strategies...")
    
    # Test different binding modalities
    strategies = [
        ('transcriptomics', "Transcriptomics as binding modality"),
        ('proteomics', "Proteomics as binding modality"),
        ('genomics', "Genomics as binding modality"),
        ('metabolomics', "Metabolomics as binding modality")
    ]
    
    results = {}
    
    for binding_mod, description in strategies:
        print(f"\n📋 Testing: {description}")
        
        # Binding modality approach (now the only approach)
        loss = contrastive_loss(embeddings, binding_modality=binding_mod)
        
        results[description] = loss.item()
        print(f"   Contrastive Loss: {loss.item():.4f}")
    
    # Show comparison
    print(f"\n📈 Loss Comparison:")
    for strategy, loss_val in results.items():
        print(f"   {strategy}: {loss_val:.4f}")
    
    return results


def demonstrate_model_with_binding():
    """Demonstrate how to use the model with binding modality."""
    print(f"\n🏗️  Model Training with Binding Modality")
    print("=" * 60)
    
    # Define input dimensions
    input_dims = {
        'transcriptomics': 5000,
        'proteomics': 2000,
        'metabolomics': 800,
        'cell_painting': 1200,
        'genomics': 10000
    }
    
    print(f"\n🔧 Model Configurations:")
    
    # Test different configurations
    configs = [
        {
            'name': 'Transcriptomics Binding',
            'binding_modality': 'transcriptomics',
            'description': 'Use transcriptomics as anchor (most comprehensive)'
        },
        {
            'name': 'Proteomics Binding', 
            'binding_modality': 'proteomics',
            'description': 'Use proteomics as anchor (functional readout)'
        },
        {
            'name': 'Genomics Binding',
            'binding_modality': 'genomics',
            'description': 'Use genomics as anchor (genetic foundation)'
        }
    ]
    
    models = {}
    
    for config in configs:
        print(f"\n📦 Creating model: {config['name']}")
        print(f"   Description: {config['description']}")
        
        model = MultiOmicsBindWithHead(
            input_dims=input_dims,
            cat_dims=[5, 3],  # 5 drugs, 3 cell lines
            num_dims=1,       # dose
            embed_dim=768,
            num_classes=3,
            binding_modality=config['binding_modality']
        )
        
        print(f"   Binding Modality: {model.get_binding_modality()}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        models[config['name']] = model
    
    # Demonstrate dynamic binding modality changes
    print(f"\n🔄 Dynamic Binding Modality Changes:")
    model = models['Transcriptomics Binding']
    
    print(f"   Initial binding modality: {model.get_binding_modality()}")
    
    model.set_binding_modality('proteomics')
    print(f"   Changed to: {model.get_binding_modality()}")
    
    model.set_binding_modality('genomics')
    print(f"   Changed to: {model.get_binding_modality()}")
    
    model.set_binding_modality('transcriptomics')
    print(f"   Reset to: {model.get_binding_modality()}")
    
    return models


def demonstrate_training_efficiency():
    """Compare training efficiency between all-pairs and binding modality approaches."""
    print(f"\n⚡ Training Efficiency Comparison")
    print("=" * 60)
    
    # Create sample embeddings with varying numbers of modalities
    modality_counts = [2, 3, 4, 5, 6]
    batch_size = 32
    embed_dim = 768
    
    efficiency_results = {
        'modality_count': [],
        'bind_mod_0_time': [],
        'bind_mod_1_time': [],
        'bind_mod_0_loss': [],
        'bind_mod_1_loss': []
    }
    
    for n_mod in modality_counts:
        print(f"\n🧪 Testing with {n_mod} modalities...")
        
        # Create sample embeddings
        embeddings = {}
        for i in range(n_mod):
            embeddings[f'modality_{i}'] = torch.randn(batch_size, embed_dim)
        
        # Time first binding modality approach
        import time
        start = time.time()
        loss_bind_1 = contrastive_loss(embeddings, binding_modality='modality_0')
        time_bind_1 = time.time() - start
        
        # Time second binding modality approach  
        start = time.time()
        loss_bind_2 = contrastive_loss(embeddings, binding_modality='modality_1' if n_mod > 1 else 'modality_0')
        time_bind_2 = time.time() - start
        
        efficiency_results['modality_count'].append(n_mod)
        efficiency_results['bind_mod_0_time'].append(time_bind_1)
        efficiency_results['bind_mod_1_time'].append(time_bind_2) 
        efficiency_results['bind_mod_0_loss'].append(loss_bind_1.item())
        efficiency_results['bind_mod_1_loss'].append(loss_bind_2.item())
        
        print(f"   Binding modality 0 time: {time_bind_1:.4f}s, loss: {loss_bind_1.item():.4f}")
        print(f"   Binding modality 1 time: {time_bind_2:.4f}s, loss: {loss_bind_2.item():.4f}")
        print(f"   Comparison ratio: {time_bind_1/time_bind_2:.2f}x" if time_bind_2 > 0 else "   Similar performance")
    
    return efficiency_results


def main():
    """Run the complete binding modality demonstration."""
    print("🔬 MultiOmicsBind: Binding Modality Implementation")
    print("Inspired by ImageBind's approach to multi-modal alignment")
    print("=" * 80)
    
    # Demonstrate the concept with synthetic data
    loss_results = demonstrate_binding_modality()
    
    # Demonstrate model usage
    models = demonstrate_model_with_binding()
    
    # Compare efficiency
    efficiency_results = demonstrate_training_efficiency()
    
    # Summary
    print(f"\n🎯 Summary and Recommendations")
    print("=" * 60)
    
    print(f"\n💡 Key Insights:")
    print(f"   • Binding modality reduces computational complexity from O(n²) to O(n)")
    print(f"   • Transcriptomics often works well as binding modality (comprehensive)")
    print(f"   • Proteomics can be effective (functional readout)")
    print(f"   • Choice depends on your specific biological question")
    
    print(f"\n🛠️  Usage Recommendations:")
    print(f"   1. Start with transcriptomics as binding modality for gene expression studies")
    print(f"   2. Use proteomics for functional studies")
    print(f"   3. Use all-pairs for exploratory analysis with few modalities")
    print(f"   4. Switch dynamically based on data availability")
    
    print(f"\n🚀 Benefits of Binding Modality Approach:")
    print(f"   • Faster training (especially with many modalities)")
    print(f"   • More stable gradients")
    print(f"   • Enables emergent cross-modal abilities")
    print(f"   • Easier to interpret relationships")
    print(f"   • Better handles missing modalities")
    
    print(f"\n✅ Implementation Complete!")
    print(f"   MultiOmicsBind now supports ImageBind-style binding modality training.")


if __name__ == "__main__":
    main()
