# Changelog

All notable changes to MultiOmicsBind will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
