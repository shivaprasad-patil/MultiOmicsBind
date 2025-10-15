# Changelog

All notable changes to MultiOmicsBind will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2025-10-15

### Added
- Automatic train/test splitting in `train_temporal_model()` with `train_split` and `test_split` parameters
- Reproducible data splits using seed=42 for consistent results
- Custom class names support in all examples and visualizations
- Feature importance visualization in complete tutorial notebook (cells 24-25)
- Test validation script (`test_examples.sh`) for automated testing
- Enhanced real-world usage demonstration in `binding_modality_example.py`

### Changed
- **Breaking**: Updated `train_temporal_model()` signature to include optional `train_split`, `test_split`, and `save_path` parameters
- Rewrote `basic_example.py` to use manual API with automatic splitting
- Updated `binding_modality_example.py` with extended real-world usage section
- Converted `flexible_modalities_example.py` to manual API approach
- Enhanced `temporal_example.py` with automatic splitting features
- Updated `MultiOmicsBind_Complete_Tutorial.ipynb` with new features

### Improved
- All examples now use proper train/test separation (no data leakage)
- Custom class names (e.g., 'Low Response', 'Medium Response', 'High Response') replace generic labels
- Consistent API usage across all examples
- Better educational value with explicit ML best practices
- Enhanced visualization with meaningful class labels
- Removed redundant UMAP display cell from tutorial notebook

### Fixed
- Removed undefined `history` variable reference in `basic_example.py`
- Fixed import statements across all examples to use correct functions
- Corrected evaluation function calls to use `evaluate_temporal_model()`

### Testing
- Validated all examples with import tests
- Completed full training test on `basic_example.py` (20 epochs, converged)
- Model performance: 3.4M parameters, 13MB saved model, loss convergence 4.18→0.20
- Generated test data: 120MB total (transcriptomics 62MB, proteomics 42MB, cell_painting 16MB)

## [0.1.2] - 2024-12-XX

### Added
- Enhanced flexibility for any number of modalities and features
- Dynamic architecture visualization that adapts to data configuration
- New flexible_modalities_example.py demonstrating scalability
- Support for custom modality types (genomics, imaging, clinical, etc.)
- Automatic handling of different feature scales (100 to 500K+ features)
- Comprehensive documentation on customizing modalities

### Changed
- Updated basic_example.py with configurable modality setup
- Enhanced plot_architecture() function with custom_modalities parameter
- Improved README with flexibility examples and use cases
- Updated all examples to show modality flexibility

### Improved
- Better error handling for different data scales
- More robust data loading for varying feature dimensions
- Enhanced visualization for multi-modal architectures

## [0.1.1] - 2024-12-XX

### Changed
- Updated license from MIT to Apache License 2.0
- Updated license references in README.md and setup.py

## [0.1.0] - 2024-09-08

### Added
- Initial release of MultiOmicsBind
- Core model architecture with modality-specific encoders
- Contrastive learning implementation using InfoNCE loss
- Support for multiple omics modalities (transcriptomics, proteomics, cell painting)
- Metadata encoder for categorical and numerical features
- Optional classification head for supervised learning
- Comprehensive data loading and preprocessing utilities
- Training utilities with support for mixed objectives
- Visualization tools for architecture, embeddings, and feature importance
- Complete documentation with examples
- Basic and advanced usage examples
- Unit tests for core functionality

### Features
- **Multi-Modal Integration**: Seamless combination of different omics data types
- **Contrastive Learning**: Self-supervised alignment of multi-modal data
- **Flexible Architecture**: Support for any combination of omics modalities
- **Interpretability**: Built-in tools for model interpretation and analysis
- **Scalability**: Efficient PyTorch implementation with GPU support
- **Extensibility**: Modular design for easy customization and extension
