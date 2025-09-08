# Contributing to MultiOmicsBind

We welcome contributions to MultiOmicsBind! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

```bash
git clone https://github.com/yourusername/MultiOmicsBind.git
cd MultiOmicsBind
pip install -e ".[dev]"
```

## Code Style

We follow Python PEP 8 style guidelines. Please ensure your code:

- Uses 4 spaces for indentation
- Has line lengths ≤ 88 characters (Black formatter)
- Includes type hints where appropriate
- Has comprehensive docstrings for all public functions and classes

### Formatting

We use Black for code formatting:

```bash
black multiomicsbind/
```

### Linting

We use flake8 for linting:

```bash
flake8 multiomicsbind/
```

## Testing

All contributions should include appropriate tests. We use pytest for testing:

```bash
pytest tests/
```

### Test Coverage

Maintain test coverage above 80%. Check coverage with:

```bash
pytest --cov=multiomicsbind tests/
```

## Documentation

- All public functions and classes should have comprehensive docstrings
- Use Google-style docstrings
- Update the README.md if you add new features
- Add examples for new functionality

### Docstring Example

```python
def train_model(model, dataloader, epochs=10):
    """
    Train a MultiOmicsBind model.
    
    Args:
        model (nn.Module): The model to train
        dataloader (DataLoader): Training data loader
        epochs (int): Number of training epochs (default: 10)
        
    Returns:
        nn.Module: The trained model
        
    Example:
        >>> model = MultiOmicsBindWithHead(input_dims)
        >>> trained_model = train_model(model, dataloader, epochs=50)
    """
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

- A clear description of the bug
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment information (Python version, PyTorch version, etc.)

### Feature Requests

When requesting features, please include:

- A clear description of the feature
- Use cases for the feature
- Proposed implementation approach (if you have one)

### Code Contributions

We welcome contributions including:

- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test improvements

## Pull Request Process

1. Ensure your branch is up to date with the main branch
2. Run all tests and ensure they pass
3. Update documentation as needed
4. Ensure your code follows the style guidelines
5. Create a pull request with a clear description of your changes

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Changelog is updated (for significant changes)
- [ ] Commit messages are clear and descriptive

## Commit Messages

Use clear, descriptive commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests when appropriate

### Examples

```
Add support for metabolomics data

- Implement MetabolomicsEncoder class
- Update MultiOmicsDataset to handle metabolomics files
- Add tests for metabolomics integration
- Update documentation with metabolomics examples

Fixes #123
```

## Branching Strategy

- `main`: Stable release branch
- `develop`: Development branch for new features
- `feature/feature-name`: Feature development branches
- `bugfix/bug-description`: Bug fix branches
- `hotfix/critical-fix`: Critical bug fixes

## Release Process

1. Update version number in `setup.py` and `__init__.py`
2. Update `CHANGELOG.md`
3. Create a release tag
4. Update documentation
5. Create GitHub release

## Questions?

If you have questions about contributing, please:

1. Check the documentation
2. Search existing issues
3. Create a new issue with the "question" label
4. Join our discussions on GitHub

Thank you for contributing to MultiOmicsBind!
