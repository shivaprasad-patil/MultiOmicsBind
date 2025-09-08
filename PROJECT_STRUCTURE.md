# MultiOmicsBind Project Structure

This directory contains the complete MultiOmicsBind package for multi-omics data integration using deep learning.

## Directory Structure

```
MultiOmicsBind/
├── multiomicsbind/              # Main package source code
│   ├── __init__.py             # Package initialization
│   ├── core/                   # Core model components
│   │   ├── __init__.py
│   │   ├── encoders.py         # Neural encoders for omics data
│   │   ├── losses.py           # Contrastive and classification losses
│   │   └── model.py            # Main MultiOmicsBind model
│   ├── data/                   # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py          # PyTorch dataset classes
│   ├── training/               # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py          # Training functions and utilities
│   └── utils/                  # Visualization and utility functions
│       ├── __init__.py
│       └── visualization.py    # Plotting and visualization tools
├── examples/                   # Usage examples and tutorials
│   ├── basic_example.py        # Basic usage demonstration
│   └── advanced_analysis.py    # Advanced analysis features
├── tests/                      # Test suite
│   └── test_basic.py          # Basic functionality tests
├── setup.py                   # Package installation configuration
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation
├── LICENSE                    # MIT license
├── CHANGELOG.md              # Version history
├── CONTRIBUTING.md           # Contribution guidelines
├── architecture.png          # Model architecture diagram
└── training_history.png      # Example training curves
```

## Quick Start

1. **Install the package**:
```bash
cd MultiOmicsBind
pip install -e .
```

2. **Run basic example**:
```bash
python examples/basic_example.py
```

3. **Run tests**:
```bash
pytest tests/
```

## Features

- 🧬 **Multi-Modal Integration**: Combine transcriptomics, proteomics, cell painting, and metadata
- 🎯 **Contrastive Learning**: Self-supervised alignment across modalities
- 📊 **Flexible Architecture**: Support for any combination of omics data
- 🔍 **Interpretability**: Built-in feature importance and embedding analysis
- ⚡ **Scalable**: Efficient PyTorch implementation with GPU support
- 🧪 **Easy to Use**: Comprehensive examples and documentation

## Package Components

### Core Components (`multiomicsbind/core/`)
- **encoders.py**: Neural network encoders for different omics modalities
- **losses.py**: Contrastive learning loss functions (InfoNCE, etc.)
- **model.py**: Main MultiOmicsBind model with optional classification head

### Data Handling (`multiomicsbind/data/`)
- **dataset.py**: PyTorch dataset for loading and preprocessing multi-omics data

### Training (`multiomicsbind/training/`)
- **trainer.py**: Training loops, evaluation, and early stopping utilities

### Utilities (`multiomicsbind/utils/`)
- **visualization.py**: Plotting functions for architecture, embeddings, and results

### Examples (`examples/`)
- **basic_example.py**: Complete workflow from data loading to model training
- **advanced_analysis.py**: Feature interpretation, UMAP visualization, cross-modal analysis

### Tests (`tests/`)
- **test_basic.py**: Unit tests for core functionality

## Generated Files

When you run the examples, the following files will be generated:
- `multiomicsbind_trained.pth`: Trained model weights
- `*.csv`: Synthetic multi-omics datasets
- `*.png`: Visualization outputs (architecture, training curves, embeddings)

## Citation

If you use MultiOmicsBind in your research, please cite our work.

## Support

- 📖 Documentation: See README.md
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions

---

*MultiOmicsBind: Bridging multi-omics data with deep learning* 🧬✨
