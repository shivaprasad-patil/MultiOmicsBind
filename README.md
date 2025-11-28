# MultiOmicsBind

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**MultiOmicsBind** is a deep learning framework for integrating multi-omics data using contrastive learning. Inspired by Meta's ImageBind, it learns unified representations across different biological modalities.

## Features

- 🎯 **Unified Embeddings** - Shared representations across omics modalities
- 🔗 **Binding Modality** - O(n) complexity vs O(n²) for traditional approaches
- 📊 **Contrastive Learning** - Self-supervised multi-modal alignment
- 🧪 **Flexible Architecture** - Any combination of omics types and metadata
- 🚀 **High-Level API** - Train models with single function calls
- ⚡ **Automatic Reproducibility** - Built-in seeding for consistent results

## Installation

```bash
git clone https://github.com/shivaprasad-patil/MultiOmicsBind.git
cd MultiOmicsBind
pip install -e .
```

**Requirements**: Python 3.8+, PyTorch 1.9+, NumPy, Pandas, scikit-learn

## Quick Start

```python
import torch
from multiomicsbind import (
    MultiOmicsDataset,
    train_multiomicsbind,
    set_seed
)

# Set seed for reproducibility
set_seed(42)

# Load your data
dataset = MultiOmicsDataset(
    data_paths={
        'transcriptomics': 'transcriptomics.csv',
        'proteomics': 'proteomics.csv',
        'metabolomics': 'metabolomics.csv'
    },
    metadata_path='metadata.csv',
    label_col='response',
    binding_modality='transcriptomics'  # Anchor modality
)

# Train model (automatically reproducible with seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, history = train_multiomicsbind(
    dataset=dataset,
    device=device,
    epochs=20,
    batch_size=32,
    embed_dim=256
)

# Extract embeddings for downstream analysis
embeddings = model.get_embeddings(dataset)
```

## Data Format

### CSV Files
Each modality requires a CSV file with:
- `sample_id` column (first column)
- Feature columns (numerical values)
- Samples in the same order across all files

```csv
sample_id,feature_1,feature_2,...,feature_N
sample_001,0.234,-1.456,...,0.891
sample_002,1.123,-0.234,...,-0.567
```

### Metadata
```csv
sample_id,response,drug,dose
sample_001,0,Drug_A,1.5
sample_002,1,Drug_B,2.0
```

### Data Normalization

MultiOmicsBind automatically normalizes data to mean=0, std=1 by default:

```python
# Recommended: Let MultiOmicsBind normalize your data
dataset = MultiOmicsDataset(
    data_paths={...},
    metadata_path='metadata.csv',
    normalize=True  # Default: applies z-score standardization
)

# If already z-score normalized
dataset = MultiOmicsDataset(
    data_paths={...},
    normalize=False  # Skip normalization
)
```

**Binary Data**: Automatically detected (e.g., mutation data with 0/1 values) and normalization is skipped.

## Binding Modality Concept

Traditional multi-omics integration compares all modality pairs (O(n²) complexity). MultiOmicsBind uses a **binding modality** approach (O(n) complexity):

| Approach | Complexity | Comparisons | Speed |
|----------|------------|-------------|-------|
| All-Pairs | O(n²) | n×(n-1)/2 | 1x |
| **Binding Modality** | **O(n)** | **n-1** | **5x+** |

Choose the most stable/comprehensive modality as your anchor (often transcriptomics).

## Temporal Data

For time-series omics data:

```python
from multiomicsbind import TemporalMultiOmicsDataset, train_temporal_model

dataset = TemporalMultiOmicsDataset(
    static_data_paths={'transcriptomics': 'transcriptomics_t0.csv'},
    temporal_data_paths={'proteomics': 'proteomics_timeseries.csv'},
    metadata_path='metadata.csv',
    temporal_metadata_path='temporal_metadata.csv',
    label_col='response',
    binding_modality='transcriptomics'
)

model, history = train_temporal_model(
    dataset=dataset,
    device=device,
    epochs=20
)
```

## Feature Importance

```python
from multiomicsbind import compute_feature_importance

importance_scores = compute_feature_importance(
    model=model,
    dataset=dataset,
    device=device,
    method='gradient'  # Gradient-based attribution
)

# Scores normalized to [0, 1] per modality
# 1.0 = most important, 0.0 = least important
```

## Examples

See the `examples/` directory for detailed tutorials:
- `basic_example.py` - Standard multi-omics integration
- `binding_modality_example.py` - Binding modality demonstration
- `temporal_example.py` - Time-series omics data
- `flexible_modalities_example.py` - Variable modality configurations
- `advanced_analysis.py` - Feature importance and embeddings

## Reproducibility

MultiOmicsBind ensures reproducibility by default:

```python
# Automatic seeding (default seed=42)
model, history = train_multiomicsbind(dataset, device, epochs=20)

# Custom seed
model, history = train_multiomicsbind(dataset, device, epochs=20, seed=123)

# Disable automatic seeding
model, history = train_multiomicsbind(dataset, device, epochs=20, seed=None)
```

## Citation

If you use MultiOmicsBind in your research, please cite:

```bibtex
@software{multiomicsbind2024,
  title={MultiOmicsBind: Deep Learning Framework for Multi-Omics Integration},
  author={Patil, Shivaprasad},
  year={2024},
  url={https://github.com/shivaprasad-patil/MultiOmicsBind}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- �� [Documentation](https://github.com/shivaprasad-patil/MultiOmicsBind)
- 🐛 [Issue Tracker](https://github.com/shivaprasad-patil/MultiOmicsBind/issues)
- 💬 [Discussions](https://github.com/shivaprasad-patil/MultiOmicsBind/discussions)
