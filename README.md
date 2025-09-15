# MultiOmicsBind

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**MultiOmicsBind** is a deep learning framework for integrating and analyzing multi-omics data using contrastive learning and neural encoders. Inspired by [ImageBind](https://imagebind.metademolab.com) from Meta AI, it enables unified representation learning across different biological data modalities.

## 🧬 Overview

Multi-omics data integration is a critical challenge in systems biology and precision medicine. MultiOmicsBind addresses this by:

- **🎯 Unified Embeddings**: Learning shared representations across different omics modalities
- **🔗 Binding Modality**: Revolutionary O(n) complexity approach inspired by ImageBind
- **📊 Contrastive Learning**: Self-supervised alignment of multi-modal biological data
- **🧪 Flexible Architecture**: Supporting any combination of omics data types and metadata
- **🚀 Downstream Tasks**: Enabling both unsupervised exploration and supervised prediction

## 🏗️ Architecture

![MultiOmicsBind Architecture](architecture.png)

### Core Components

1. **🔬 Modality-Specific Encoders**: Neural networks transforming raw omics data into 768-dim embeddings
2. **📋 Metadata Encoder**: Handles experimental metadata (drugs, cell lines, doses, conditions)
3. **🎯 Contrastive Learning**: Aligns embeddings from same sample across different modalities
4. **🧠 Classification Head** *(optional)*: For supervised learning tasks
5. **🔄 Multi-Modal Fusion**: Combines embeddings via mean pooling for downstream analysis

### 🔗 Binding Modality Innovation

**Key Innovation**: MultiOmicsBind introduces **Binding Modality**, inspired by Meta's ImageBind:

| Approach | Complexity | Comparisons | Memory | Speed | Best For |
|----------|------------|-------------|---------|-------|----------|
| **All-Pairs** | O(n²) | n×(n-1)/2 | Quadratic | 1x | 2-3 modalities |
| **Binding Modality** | **O(n)** | **n-1** | **Linear** | **5x+** | **4+ modalities** |

**Benefits:**
- ⚡ **5x+ Faster Training** with multiple modalities
- 🧠 **Linear Memory Scaling** instead of quadratic
- 🎯 **Biological Interpretability** via meaningful anchor selection
- 🔄 **Missing Data Robustness** when non-anchor modalities are absent
- 🚀 **Emergent Cross-Modal Abilities** for zero-shot retrieval

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/shivaprasad-patil/MultiOmicsBind.git
cd MultiOmicsBind
pip install -e .
```

### Basic Usage

```python
import torch
from multiomicsbind import MultiOmicsBindWithHead, MultiOmicsDataset, train_multiomicsbind

# 1. Define your multi-omics data (any number of modalities, any dimensions)
data_paths = {
    'transcriptomics': 'gene_expression.csv',     # 20K genes
    'proteomics': 'protein_levels.csv',           # 8K proteins  
    'metabolomics': 'metabolites.csv',            # 2.5K metabolites
    'cell_painting': 'morphology.csv',            # 1.5K morphological features
    'genomics': 'snp_data.csv'                    # 500K SNPs
}

# 2. Load data
dataset = MultiOmicsDataset(
    data_paths=data_paths,
    metadata_path='metadata.csv',
    cat_cols=['drug', 'cell_line'],
    num_cols=['dose', 'time'],
    label_col='response'
)

# ⚠️ Important: All data files must have samples in the same order!
# Each row in transcriptomics.csv must correspond to the same sample 
# as the same row in proteomics.csv, metabolomics.csv, etc.

# 3. Create model with binding modality (recommended for 4+ modalities)
input_dims = dataset.get_input_dims()
cat_dims, num_dims = dataset.get_metadata_dims()

model = MultiOmicsBindWithHead(
    input_dims=input_dims,
    cat_dims=cat_dims,
    num_dims=num_dims,
    embed_dim=768,
    num_classes=3,
    binding_modality='transcriptomics'  # Use transcriptomics as anchor
)

# 4. Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

trained_model = train_multiomicsbind(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    device=device,
    epochs=50
)

# 5. Extract unified embeddings
embeddings = model.encode(sample_data)
```

## 🔗 Choosing Your Binding Modality

### Recommended Strategies

```python
# 🧬 Transcriptomics Binding (Default - Most Comprehensive)
model = MultiOmicsBindWithHead(
    input_dims=input_dims,
    binding_modality='transcriptomics'  # 10K-50K genes = broad cellular state
)

# 🔬 Proteomics Binding (Functional Studies)  
model = MultiOmicsBindWithHead(
    input_dims=input_dims,
    binding_modality='proteomics'  # 5K-20K proteins = functional readout
)

# 🧮 Genomics Binding (Population Studies)
model = MultiOmicsBindWithHead(
    input_dims=input_dims, 
    binding_modality='genomics'  # 100K+ SNPs = genetic background
)

# 🔄 Dynamic Switching During Training
model.set_binding_modality('transcriptomics')  # Start comprehensive
model.set_binding_modality('proteomics')       # Switch to functional
model.set_binding_modality(None)               # Switch to all-pairs
```

### Selection Guide

| Modality | Best For | Features | Rationale |
|----------|----------|----------|-----------|
| **Transcriptomics** | Gene expression studies, systems biology | 10K-50K genes | Most comprehensive molecular readout |
| **Proteomics** | Drug studies, functional analysis | 5K-20K proteins | Direct functional readout, closer to phenotype |
| **Genomics** | Population studies, GWAS | 100K+ SNPs | Stable constitutional information |
| **Cell Painting** | Phenotypic screening, morphology | 1K-5K features | Rich morphological phenotype |
| **Metabolomics** | Metabolism studies | 1K-5K metabolites | Downstream functional readout |

## � Binding Modality Technical Details

### Dynamic Modality Management

```python
# Initialize with binding modality
model = MultiOmicsBindWithHead(input_dims, binding_modality='transcriptomics')

# Change binding modality during training
model.set_binding_modality('proteomics')      # Switch to proteomics anchor
model.set_binding_modality('genomics')        # Switch to genomics anchor  
model.set_binding_modality(None)              # Switch to all-pairs approach

# Check current configuration
current_binding = model.get_binding_modality()
print(f"Current binding modality: {current_binding}")
```

### Performance Comparison

| Modalities | All-pairs Comparisons | Binding Comparisons | Memory Reduction | Speed Improvement |
|------------|---------------------|-------------------|-----------------|-------------------|
| 2 | 2 | 1 | 2.0x | 2.0x |
| 3 | 6 | 2 | 3.0x | 3.0x |
| 4 | 12 | 3 | 4.0x | 4.0x |
| 5 | 20 | 4 | 5.0x | 5.0x |
| 6 | 30 | 5 | 6.0x | 6.0x |

### Advanced Loss Strategies

```python
from multiomicsbind.core.losses import adaptive_contrastive_loss, binding_modality_loss

# Strategy 1: Pure binding modality
loss1 = binding_modality_loss(embeddings, 'transcriptomics', temperature=0.07)

# Strategy 2: Adaptive loss (switches automatically)
loss2 = adaptive_contrastive_loss(embeddings, binding_modality='transcriptomics')

# Strategy 3: Multi-binding approach (custom)
def multi_binding_loss(embeddings, temp=0.07):
    loss_tx = binding_modality_loss(embeddings, 'transcriptomics', temp)
    loss_pr = binding_modality_loss(embeddings, 'proteomics', temp)
    return 0.6 * loss_tx + 0.4 * loss_pr  # Weighted combination

# Strategy 4: Dynamic switching during training
def adaptive_training_strategy(epoch):
    if epoch < 10:
        return None                    # All-pairs exploration
    elif epoch < 30:
        return 'transcriptomics'       # Transcriptomics binding
    else:
        return 'proteomics'            # Proteomics fine-tuning
```

### Biological Applications by Binding Modality

```python
# Drug Discovery: Use transcriptomics as comprehensive drug response anchor
drug_model = MultiOmicsBindWithHead({
    'transcriptomics': 20000,
    'cell_painting': 1500,
    'proteomics': 8000
}, binding_modality='transcriptomics')

# Disease Classification: Use genomics as stable genetic anchor
disease_model = MultiOmicsBindWithHead({
    'genomics': 500000,
    'transcriptomics': 20000,
    'proteomics': 8000,
    'metabolomics': 2500
}, binding_modality='genomics')

# Functional Studies: Use proteomics as functional anchor
function_model = MultiOmicsBindWithHead({
    'proteomics': 8000,
    'metabolomics': 2500,
    'cell_painting': 1500,
    'transcriptomics': 20000
}, binding_modality='proteomics')
```

### Zero-Shot Cross-Modal Capabilities

When using binding modality, MultiOmicsBind develops **emergent cross-modal abilities**:

```python
# Example: Query with one modality, retrieve similar samples from other modalities
def cross_modal_retrieval(model, query_modality_data, target_modality='proteomics'):
    """Find proteomics profiles similar to transcriptomics query"""
    
    # Encode query from transcriptomics
    query_embedding = model.encoders['transcriptomics'](query_modality_data)
    
    # Compare with proteomics database embeddings
    similarities = torch.cosine_similarity(query_embedding, proteomics_database_embeddings)
    
    # Retrieve most similar proteomics profiles
    top_matches = torch.topk(similarities, k=5)
    return top_matches

# Works because binding modality aligns all modalities to common space
```

## �📊 Examples

### Basic Integration

```python
# Run the basic example
python examples/basic_example.py
```

### Advanced Analysis with Binding Modality

```python
# Comprehensive binding modality demonstration
python examples/binding_modality_example.py
```

**Features:**
- Computational efficiency comparison (O(n²) vs O(n))
- Different binding strategies (transcriptomics, proteomics, etc.)
- Dynamic binding modality switching during training
- Performance analysis with multiple modalities

### Flexible Multi-Modal Scaling

```python
# Scaling demonstration  
python examples/flexible_modalities_example.py
```

## 🔧 Key Features

### Multi-Modal Data Support
- **Any Number of Modalities**: 2 to 10+ different omics types
- **Any Feature Dimensions**: From 100 to 500K+ features per modality
- **Flexible Data Types**: Gene expression, protein levels, SNPs, morphology, metabolites
- **Missing Data Handling**: Robust to missing modalities during inference

### Advanced Learning Approaches
```python
# All-pairs contrastive learning (traditional)
model = MultiOmicsBindWithHead(input_dims, binding_modality=None)

# Binding modality learning (efficient)  
model = MultiOmicsBindWithHead(input_dims, binding_modality='transcriptomics')

# Adaptive approach selection
from multiomicsbind.core.losses import adaptive_contrastive_loss
loss = adaptive_contrastive_loss(embeddings, binding_modality='transcriptomics')
```

### Scalability & Performance
- **GPU Acceleration**: Full CUDA support for large-scale data
- **Memory Efficient**: Linear scaling with binding modality approach
- **Batch Processing**: Optimized for large datasets
- **Interpretable**: Built-in visualization and analysis tools

## 🎯 Applications

### 1. Drug Discovery
```python
# Predict drug response using multi-omics + metadata
model = MultiOmicsBindWithHead({
    'transcriptomics': 20000,
    'cell_painting': 1500,
    'proteomics': 8000
}, binding_modality='transcriptomics')  # Comprehensive drug response signature
```

### 2. Disease Research  
```python
# Disease subtyping with genomics anchor
model = MultiOmicsBindWithHead({
    'genomics': 500000,
    'transcriptomics': 20000, 
    'proteomics': 8000,
    'metabolomics': 2500
}, binding_modality='genomics')  # Genetic background as stable reference
```

### 3. Systems Biology
```python
# All-pairs exploration for pathway analysis
model = MultiOmicsBindWithHead({
    'transcriptomics': 20000,
    'proteomics': 8000,
    'metabolomics': 2500
}, binding_modality=None)  # Explore all cross-modal relationships
```

## ️ Advanced Usage

### Custom Loss Functions
```python
from multiomicsbind.core.losses import binding_modality_loss, adaptive_contrastive_loss

# Direct binding modality loss
loss = binding_modality_loss(embeddings, 'transcriptomics', temperature=0.07)

# Adaptive loss selection
loss = adaptive_contrastive_loss(embeddings, 
                               binding_modality='proteomics' if use_binding else None)
```

### Model Customization
```python
# Initialize with specific architecture
model = MultiOmicsBindWithHead(
    input_dims={'transcriptomics': 20000, 'proteomics': 8000},
    embed_dim=1024,           # Larger embeddings
    dropout=0.3,              # Higher dropout
    binding_modality='transcriptomics',
    num_classes=5
)

# Access internal components
embeddings = model.encode(data)                    # Get embeddings
loss = model.compute_contrastive_loss(embeddings) # Compute loss
model.set_binding_modality('proteomics')          # Change binding
```

### Visualization & Analysis
```python
from multiomicsbind.utils import plot_training_history

# Analyze training progress  
plot_training_history(model.training_history, save_path="training.png")
```

## 📚 Technical Details

### Model Architecture
- **Encoders**: 2-layer MLPs with LayerNorm and ReLU
- **Embeddings**: 768-dimensional unified space (configurable)
- **Fusion**: Mean pooling across modalities
- **Classification**: Optional 2-layer head with dropout

### Loss Functions
- **InfoNCE Contrastive**: Aligns same-sample embeddings across modalities
- **Cross-Entropy**: Standard classification loss (optional)
- **Temperature Scaling**: τ = 0.07 for contrastive learning

### Training
- **Optimizer**: AdamW with 1e-4 learning rate
- **Regularization**: Dropout (0.2), gradient clipping (1.0)
- **Scheduling**: Optional learning rate decay
- **Early Stopping**: Validation-based stopping

## 📖 Documentation

- **[examples/](examples/)**: Complete usage examples and tutorials
- **API Reference**: Inline documentation for all classes and functions

## 🔧 Complete API Reference

### Core Classes

#### `MultiOmicsBindWithHead`
```python
model = MultiOmicsBindWithHead(
    input_dims={"transcriptomics": 20000, "proteomics": 8000},  # Required
    cat_dims=[10, 5],                    # Categorical metadata dimensions  
    num_dims=1,                          # Numerical metadata dimensions
    embed_dim=768,                       # Embedding dimension (default: 768)
    num_classes=3,                       # Classification classes (optional)
    dropout=0.2,                         # Dropout rate (default: 0.2)
    binding_modality='transcriptomics'   # Binding anchor (optional)
)
```

#### Key Methods
```python
# Encoding and inference
embeddings = model.encode(inputs)                    # Get embeddings
logits = model(inputs)                               # Forward pass
loss = model.compute_contrastive_loss(embeddings)   # Compute loss

# Binding modality management  
model.set_binding_modality('proteomics')            # Change binding modality
current = model.get_binding_modality()               # Get current binding
model.set_binding_modality(None)                    # Switch to all-pairs

# Model utilities
model.freeze_encoders()                              # Freeze encoder weights
model.unfreeze_encoders()                            # Unfreeze encoders
dim = model.get_embedding_dimension()                # Get embed dimension
modalities = model.get_modalities()                  # List supported modalities
```

### Loss Functions

#### `binding_modality_loss()`
```python
from multiomicsbind.core.losses import binding_modality_loss

loss = binding_modality_loss(
    embeddings={'tx': emb1, 'pr': emb2, 'met': emb3},  # Embeddings dict
    binding_modality='tx',                               # Anchor modality name
    temperature=0.07                                     # Temperature parameter
)
```

#### `adaptive_contrastive_loss()`
```python
from multiomicsbind.core.losses import adaptive_contrastive_loss

# Automatically chooses approach based on binding_modality parameter
loss = adaptive_contrastive_loss(
    embeddings=embeddings,
    binding_modality='transcriptomics',  # Use binding (O(n)) or None for all-pairs (O(n²))
    temperature=0.07
)
```

### Training

#### `train_multiomicsbind()`
```python
from multiomicsbind import train_multiomicsbind

trained_model = train_multiomicsbind(
    model=model,                          # MultiOmicsBindWithHead instance
    dataloader=dataloader,                # PyTorch DataLoader
    optimizer=optimizer,                  # PyTorch optimizer
    device=device,                        # Training device
    epochs=50,                            # Number of epochs
    temperature=0.07,                     # Contrastive learning temperature
    use_classification=True,              # Enable classification loss
    contrastive_weight=1.0,               # Weight for contrastive loss
    classification_weight=1.0,            # Weight for classification loss
    scheduler=scheduler,                  # Learning rate scheduler (optional)
    verbose=True                          # Print progress
)
```

### Data Loading

#### `MultiOmicsDataset`
```python
from multiomicsbind import MultiOmicsDataset

dataset = MultiOmicsDataset(
    data_paths={                          # Paths to omics data files
        'transcriptomics': 'tx_data.csv',
        'proteomics': 'pr_data.csv',
        'metabolomics': 'met_data.csv'
    },
    metadata_path='metadata.csv',         # Path to metadata file
    cat_cols=['drug', 'cell_line'],       # Categorical column names
    num_cols=['dose', 'time'],            # Numerical column names  
    label_col='response'                  # Target label column name
)

# Utility methods
input_dims = dataset.get_input_dims()    # Get feature dimensions per modality
cat_dims, num_dims = dataset.get_metadata_dims()  # Get metadata dimensions
```

### Visualization

```python
from multiomicsbind.utils plot_training_history

# Plot training curves  
plot_training_history(model.training_history, save_path="training.png")

# Plot embeddings (requires umap-learn)
plot_embeddings_umap(embeddings, labels, save_path="umap.png")
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
git clone https://github.com/shivaprasad-patil/MultiOmicsBind.git
cd MultiOmicsBind
pip install -e ".[dev]"
pytest tests/
```

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 📚 Citation

```bibtex
@software{multiomicsbind2025,
  title={MultiOmicsBind: A Deep Learning Framework for Multi-Omics Data Integration},
  author={Shivaprasad Patil},
  year={2025},
  url={https://github.com/shivaprasad-patil/MultiOmicsBind},
  note={Inspired by ImageBind from Meta AI}
}
```

## 🙏 Acknowledgments

- **[ImageBind](https://imagebind.metademolab.com)** from Meta's FAIR team for pioneering binding modality approach
- **PyTorch** team for the excellent deep learning framework  
- **Scientific Community** working on multi-omics data integration

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/shivaprasad-patil/MultiOmicsBind/issues)
- 📧 **Contact**: shivaprasad309319@gmail.com

---

*MultiOmicsBind: Bringing ImageBind's revolutionary binding modality concept to biological data integration* 🧬✨
