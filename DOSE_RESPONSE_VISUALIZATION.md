# Dose-Response Visualization in MultiOmicsBind

## Overview

This document describes the new dose-response visualization feature that shows how dose values contribute to predictions and how they correlate with different response classes.

## New Function: `plot_dose_response_analysis()`

### Location
`multiomicsbind/utils/visualization.py`

### Purpose
Visualize dose-response relationships with three complementary plots:
1. **Dose Distribution by Class** - Violin plots showing dose ranges for each response class
2. **Dose vs Predictions** - Scatter plot showing how dose correlates with prediction accuracy
3. **Mean Dose Comparison** - Bar chart comparing mean doses between true and predicted classes

### Function Signature

```python
def plot_dose_response_analysis(
    doses: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6)
) -> None
```

### Parameters

- **doses** (`np.ndarray`): Dose values for each sample
- **labels** (`np.ndarray`): True labels/response classes
- **predictions** (`np.ndarray`): Model predictions
- **class_names** (`Optional[List[str]]`): Custom names for each class (e.g., ['No Response', 'Partial Response', 'Full Response'])
- **save_path** (`Optional[str]`): Path to save the figure
- **figsize** (`Tuple[int, int]`): Figure size (default: 16x6 inches)

## Usage Example

### Basic Usage

```python
from multiomicsbind.utils.visualization import plot_dose_response_analysis
import numpy as np

# Your data
doses = np.array([1.5, 2.0, 5.5, 7.0, ...])  # Dose values
labels = np.array([0, 0, 1, 2, ...])         # True classes
predictions = np.array([0, 0, 1, 2, ...])    # Predicted classes

# Define meaningful class names
class_names = ['No Response', 'Partial Response', 'Full Response']

# Generate visualization
plot_dose_response_analysis(
    doses=doses,
    labels=labels,
    predictions=predictions,
    class_names=class_names,
    save_path='dose_response_analysis.png'
)
```

### Complete Workflow Example

```python
import pandas as pd
from multiomicsbind.utils.visualization import plot_dose_response_analysis

# Load your metadata with dose information
metadata = pd.read_csv('metadata.csv')

# Get test set metadata (assuming you have test_dataset.indices)
test_metadata = metadata.iloc[test_dataset.indices].reset_index(drop=True)

# Extract values
doses = test_metadata['dose'].values
true_labels = report['labels']
predictions = report['predictions']

# Define class names
class_names = ['No Response', 'Partial Response', 'Full Response']

# Generate visualization
plot_dose_response_analysis(
    doses=doses,
    labels=true_labels,
    predictions=predictions,
    class_names=class_names,
    save_path='dose_response_analysis.png'
)
```

## Interpretation Guide

### Plot 1: Dose Distribution by True Class (Violin Plot)
- **What it shows**: The distribution of dose values for each response class
- **How to interpret**:
  - Width of violin = density of samples at that dose level
  - White dot = median dose
  - Thick bar = interquartile range (25th-75th percentile)
  - Thin line = min/max range
- **What to look for**:
  - Clear separation = good dose-response relationship
  - Overlapping distributions = dose alone may not predict response
  - Multimodal distributions = potential subgroups within response class

### Plot 2: Dose vs Predictions (Scatter Plot)
- **What it shows**: How dose relates to predictions, highlighting correct vs incorrect predictions
- **How to interpret**:
  - Color-coded points = predicted class
  - Circle (o) = correct prediction
  - X mark = incorrect prediction
- **What to look for**:
  - Horizontal bands = model successfully learned dose thresholds
  - Many X marks = poor dose-response learning
  - Random scatter = dose not predictive for this task

### Plot 3: Mean Dose per Class (Bar Chart with Error Bars)
- **What it shows**: Comparison of mean doses between true and predicted classes
- **How to interpret**:
  - Blue bars = true class mean doses
  - Coral bars = predicted class mean doses
  - Error bars = standard deviation
  - Matching bars = perfect prediction for that class
- **What to look for**:
  - Similar heights = model correctly identifies dose ranges
  - Different heights = systematic prediction bias
  - Large error bars = high variability within class

## Real-World Use Cases

### 1. Drug Screening
**Scenario**: Testing compounds at different concentrations to determine IC50

```python
class_names = ['No Inhibition', 'Partial Inhibition', 'Full Inhibition']
```

**Interpretation**:
- Low doses → No Inhibition
- Mid doses → Partial Inhibition
- High doses → Full Inhibition

### 2. Cell Viability Assays
**Scenario**: Measuring cell death at various treatment doses

```python
class_names = ['Viable', 'Damaged', 'Dead']
```

**Interpretation**: Clear dose-response indicates treatment efficacy

### 3. Cytokine Response
**Scenario**: Measuring immune response at different stimulation levels

```python
class_names = ['No Response', 'Moderate Response', 'Strong Response']
```

**Interpretation**: Can identify optimal stimulation dose

### 4. Pharmacokinetics
**Scenario**: Drug concentration vs therapeutic effect

```python
class_names = ['Sub-therapeutic', 'Therapeutic Window', 'Toxic']
```

**Interpretation**: Helps identify safe and effective dose range

## Integration with MultiOmicsBind Workflow

The dose-response analysis is fully integrated with the MultiOmicsBind workflow:

```python
# 1. Create dataset with dose as numerical metadata
dataset = TemporalMultiOmicsDataset(
    # ... data files ...
    metadata_file='metadata.csv',
    label_col='response',
    num_cols=['dose'],  # ✅ Include dose as numerical metadata
)

# 2. Train model (dose contributes to predictions)
model, history = train_temporal_model(
    dataset=train_dataset,
    # ... training parameters ...
)

# 3. Evaluate and get predictions
report = create_analysis_report(
    model=model,
    dataset=test_dataset,
    class_names=['No Response', 'Partial Response', 'Full Response'],
    # ...
)

# 4. Visualize dose-response relationship
metadata = pd.read_csv('metadata.csv')
test_metadata = metadata.iloc[test_dataset.indices]

plot_dose_response_analysis(
    doses=test_metadata['dose'].values,
    labels=report['labels'],
    predictions=report['predictions'],
    class_names=['No Response', 'Partial Response', 'Full Response'],
    save_path='dose_response.png'
)
```

## Key Features

### ✅ Automatic Class Name Handling
- Falls back to "Class 0", "Class 1", etc. if no names provided
- Supports any number of classes

### ✅ Publication-Quality Plots
- High resolution (300 DPI)
- Professional styling with seaborn/matplotlib
- Color-coded for clarity
- Grid lines for easy reading

### ✅ Statistical Visualization
- Violin plots show full distribution
- Error bars show variability
- Mean and median clearly marked

### ✅ Prediction Accuracy Highlighted
- Correct predictions marked with circles
- Incorrect predictions marked with X
- Easy identification of prediction errors

## Technical Notes

### Handling Missing Data
The function handles edge cases:
- Empty classes (no predictions for that class)
- Single-sample classes (no std deviation)
- NaN values are automatically excluded

### Color Scheme
- Uses matplotlib's viridis colormap for classes (colorblind-friendly)
- Red X marks for incorrect predictions (universally recognizable)
- Blue/coral bars for comparison (high contrast)

### Performance
- Fast for typical datasets (< 1 second for 1000 samples)
- Memory efficient (processes in single pass)
- No GPU required (uses numpy/matplotlib)

## Related Functions

### Class Names in UMAP Plots
The same `class_names` parameter works throughout the codebase:

```python
# In create_analysis_report()
report = create_analysis_report(
    model=model,
    dataset=test_dataset,
    class_names=['No Response', 'Partial Response', 'Full Response'],
    # UMAP plots will now show "No Response" instead of "Class 0"
)

# In plot_embeddings_umap()
plot_embeddings_umap(
    embeddings=embeddings,
    labels=labels,
    class_names=['No Response', 'Partial Response', 'Full Response'],
    save_path='umap.png'
)

# In plot_confusion_matrix()
plot_confusion_matrix(
    true_labels=labels,
    predicted_labels=predictions,
    class_names=['No Response', 'Partial Response', 'Full Response'],
    save_path='confusion.png'
)
```

## Example Output

The function generates a single figure with three subplots:

```
dose_response_analysis.png
├── Left: Dose Distribution by True Class (Violin plot)
├── Center: Dose vs Predictions (Scatter plot with accuracy markers)
└── Right: Mean Dose per Class ± SD (Bar chart comparison)
```

## Best Practices

### 1. Always Use Meaningful Class Names
❌ Bad: `class_names = ['Class 0', 'Class 1', 'Class 2']`  
✅ Good: `class_names = ['No Response', 'Partial Response', 'Full Response']`

### 2. Include Dose as Numerical Metadata
❌ Bad: Using sample names like "sample1_dose1", "sample1_dose2"  
✅ Good: Including dose column in metadata with `num_cols=['dose']`

### 3. Always Evaluate on Test Set
❌ Bad: Plotting training set dose-response  
✅ Good: Plotting test set dose-response (unseen data)

### 4. Document Dose Units
Always include units in your axis labels and documentation (e.g., μM, nM, mg/kg)

## Troubleshooting

### Issue: All predictions in one class
**Cause**: Model hasn't learned dose-response relationship  
**Solution**: 
- Check dose distribution is balanced
- Ensure dose included in `num_cols`
- Try longer training
- Check for data leakage

### Issue: Plots show "Class 0, 1, 2" instead of names
**Cause**: `class_names` parameter not passed or incorrect  
**Solution**: 
- Ensure class_names list matches label values: `class_names[i]` → label `i`
- Pass to both `create_analysis_report()` and `plot_dose_response_analysis()`

### Issue: Error bars too large
**Cause**: High variability in dose within each class  
**Solution**: This is informative! It shows dose ranges overlap between classes.

## Citation

If you use this visualization in your research, please cite:

```bibtex
@software{multiomicsbind2024,
  author = {Shivaprasad Patil},
  title = {MultiOmicsBind: Integrative Multi-Omics Analysis with Binding Modality},
  year = {2024},
  url = {https://github.com/shivaprasad-patil/MultiOmicsBind}
}
```

## See Also

- [ADVANCED_USAGE_GUIDE.md](ADVANCED_USAGE_GUIDE.md) - Section 4: Dose-Response Relationships
- [QUICK_ANSWERS.md](QUICK_ANSWERS.md) - Q4: How to handle dose-response data
- [temporal_example.py](examples/temporal_example.py) - Complete working example
