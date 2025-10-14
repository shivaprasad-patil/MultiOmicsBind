# Advanced Usage Guide

## Table of Contents
1. [Using Class Names in UMAP Visualizations](#1-using-class-names-in-umap-visualizations)
2. [Data Leakage Prevention](#2-data-leakage-prevention)
3. [Temporal Data Contribution](#3-temporal-data-contribution-to-predictions)
4. [Handling Dose-Response Relationships](#4-handling-dose-response-relationships)

---

## 1. Using Class Names in UMAP Visualizations

### Problem
By default, UMAPs show generic labels like "Class 0", "Class 1", "Class 2". You want to see meaningful names like "No Response", "Partial Response", "Full Response".

### Solution

The `create_analysis_report()` function supports a `class_names` parameter:

```python
from multiomicsbind import create_analysis_report

# Define your class names based on label_col values
class_names = ['No Response', 'Partial Response', 'Full Response']

# Generate report with custom class names
report = create_analysis_report(
    model=model,
    dataset=test_dataset,
    device=device,
    class_names=class_names,  # ✅ Add this parameter
    output_dir='./results',
    verbose=True
)
```

### How It Works

The `class_names` list should map to your label values:
- `class_names[0]` → label value 0
- `class_names[1]` → label value 1
- `class_names[2]` → label value 2

### Example: Metadata with String Labels

If your metadata has string labels:

```python
# metadata.csv
# sample_id,drug,dose,response
# sample_001,Drug_A,1.0,Responder
# sample_002,Drug_B,5.0,Non-responder
# ...

# When creating dataset, labels are converted to integers
dataset = TemporalMultiOmicsDataset(
    static_data_paths={'transcriptomics': 'genes.csv'},
    metadata_path='metadata.csv',
    label_col='response',  # Contains: 'Responder', 'Non-responder'
    normalize=True
)

# Get unique labels and create mapping
import pandas as pd
metadata = pd.read_csv('metadata.csv')
unique_labels = sorted(metadata['response'].unique())
# unique_labels = ['Non-responder', 'Responder']  # Sorted alphabetically

# Labels are encoded as: Non-responder=0, Responder=1
class_names = unique_labels  # ['Non-responder', 'Responder']

# Use in analysis
report = create_analysis_report(
    model=model,
    dataset=test_dataset,
    device=device,
    class_names=class_names,  # Will show 'Non-responder' instead of 'Class 0'
    output_dir='./results'
)
```

### For Individual UMAP Plots

You can also use `class_names` when calling `plot_embeddings_umap()` directly:

```python
from multiomicsbind import plot_embeddings_umap

# After evaluation
embeddings, labels, predictions = evaluate_temporal_model(model, test_dataset, device)

# Plot with custom class names
class_names = ['No Response', 'Partial Response', 'Full Response']

plot_embeddings_umap(
    embeddings=embeddings['transcriptomics'],
    labels=labels,
    class_names=class_names,  # ✅ Custom names
    title='Transcriptomics Embeddings',
    save_path='transcriptomics_umap.png'
)
```

---

## 2. Data Leakage Prevention

### Problem
Ensuring proper train/test splits across all implementations to avoid data leakage.

### Verification Checklist

#### ✅ Correct Implementation (No Leakage)

```python
from torch.utils.data import random_split

# 1. Load full dataset
dataset = TemporalMultiOmicsDataset(...)

# 2. Fix NaN values BEFORE splitting
dataset, nan_summary = check_and_fix_all_nan_values(dataset)

# 3. Split into train and test
torch.manual_seed(42)  # For reproducibility
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 4. Train on train_dataset only
model, history = train_temporal_model(
    dataset=train_dataset,  # ✅ Only training data
    device=device,
    epochs=20,
    val_split=0.2  # Will further split train into train/val (80%/20%)
)

# 5. Evaluate on test_dataset only
embeddings, labels, predictions = evaluate_temporal_model(
    model=model,
    dataset=test_dataset,  # ✅ Held-out test set
    device=device
)

# 6. Analysis report on test set
report = create_analysis_report(
    model=model,
    dataset=test_dataset,  # ✅ Same test set
    device=device,
    output_dir='./results'
)

print(f"Test Accuracy: {(predictions == labels).mean():.4f}")
```

#### ❌ Incorrect Implementation (Data Leakage)

```python
# DON'T DO THIS!
dataset = TemporalMultiOmicsDataset(...)

# Training on full dataset
model, history = train_temporal_model(
    dataset=dataset,  # ❌ Full dataset
    device=device,
    epochs=20
)

# Evaluating on same dataset
report = create_analysis_report(
    model=model,
    dataset=dataset,  # ❌ Same data used for training!
    device=device
)
# This will show inflated accuracy (possibly 100%)
```

### Best Practice: 3-Way Split

For rigorous evaluation, use train/validation/test split:

```python
# Split into 70% train, 15% validation, 15% test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# Train with validation monitoring
model, history = train_temporal_model(
    dataset=train_dataset,
    device=device,
    epochs=50,
    val_split=0.0  # Don't split further, we already have val_dataset
)

# Manual validation loop for hyperparameter tuning
# ... tune based on val_dataset performance ...

# Final evaluation on test set (only once!)
report = create_analysis_report(
    model=model,
    dataset=test_dataset,  # Final held-out test set
    device=device
)
```

### Verifying No Leakage

Add these checks to your code:

```python
# After splitting
train_indices = set(train_dataset.indices)
test_indices = set(test_dataset.indices)

# Verify no overlap
assert len(train_indices & test_indices) == 0, "Data leakage detected!"
print(f"✓ No overlap between train ({len(train_indices)}) and test ({len(test_indices)}) sets")

# Verify coverage
assert len(train_indices) + len(test_indices) == len(dataset), "Missing samples!"
print(f"✓ All {len(dataset)} samples accounted for")
```

---

## 3. Temporal Data Contribution to Predictions

### Question
"In the temporal case, does proteomics data from different timepoints contribute more to predictions than static transcriptomics at one timepoint?"

### Answer: It Depends on the Signal

MultiOmicsBind treats each modality **equally in the loss function**, regardless of whether it's static or temporal. However, the **information content** of each modality determines its practical contribution.

### How Temporal Encoders Work

```python
# Static modality (single timepoint)
Transcriptomics:  [batch, 20000 genes] → MLP → [batch, 768]

# Temporal modality (multiple timepoints)
Proteomics:       [batch, 5_timepoints, 4000 proteins] → LSTM → [batch, 768]
                          ↑
                  Captures temporal dynamics:
                  - Early vs late response
                  - Transient vs sustained changes  
                  - Dynamic trajectories
```

### Information Content Comparison

| Modality | Data Points | Information Type | Typical Contribution |
|----------|-------------|------------------|---------------------|
| **Transcriptomics (static)** | 20,000 genes × 1 timepoint | Baseline state | Captures initial molecular signature |
| **Proteomics (temporal)** | 4,000 proteins × 5 timepoints | Dynamic trajectories | Captures response dynamics over time |

### When Temporal Data Dominates

Temporal data will contribute more when:

1. **Response is time-dependent**
   ```python
   # Early responders vs late responders
   # Temporal proteomics captures this, static transcriptomics doesn't
   ```

2. **Transient responses matter**
   ```python
   # Peak at 2h, returns to baseline by 8h
   # LSTM can capture this pattern
   ```

3. **More timepoints = more information**
   ```python
   # 4000 proteins × 5 timepoints = 20,000 measurements
   # vs 20,000 genes × 1 timepoint = 20,000 measurements
   # Similar total data, but temporal adds dynamic information
   ```

### When Static Data Dominates

Static data will contribute more when:

1. **Baseline predicts outcome**
   ```python
   # Some samples resistant from the start
   # Transcriptomics captures this pre-treatment state
   ```

2. **Higher feature dimensionality**
   ```python
   # 20,000 genes vs 4,000 proteins
   # More features → potentially more discriminative patterns
   ```

3. **Time-invariant biology**
   ```python
   # Genetic background, cell type markers
   # These don't change over time
   ```

### Model Architecture: Equal Treatment

```python
# Both modalities get equal weight in loss
loss_transcriptomics = contrastive_loss(emb_tx, emb_binding_modality)
loss_proteomics = contrastive_loss(emb_proteomics, emb_binding_modality)

total_loss = loss_transcriptomics + loss_proteomics  # Equal weights!
```

### Analyzing Actual Contribution

Use **feature importance analysis** to determine which modality contributes more:

```python
# Compute feature importance
importance_dict, importance_df = compute_feature_importance(
    model=model,
    dataset=test_dataset,
    device=device,
    n_batches=10
)

# Check modality-level importance
modality_importance = importance_df.groupby('modality')['importance'].sum()
print("\nModality Contributions:")
print(modality_importance)

# Example output:
# modality
# transcriptomics    42.3
# proteomics         57.7  ← Temporal data contributing more
# Name: importance, dtype: float64
```

### Practical Recommendation

**If temporal dynamics are biologically relevant to your question**, temporal data will naturally contribute more to predictions because it contains **richer information** (static + dynamic). This is by design and reflects the underlying biology, not a model artifact.

**To balance contributions** (if desired):

1. **Use more timepoints** for all modalities if possible
2. **Use binding modality** strategically:
   ```python
   # Make static transcriptomics the binding modality
   # This ensures all other modalities align to it
   model = TemporalMultiOmicsBind(
       static_input_dims={'transcriptomics': 20000},
       temporal_input_dims={'proteomics': 4000},
       binding_modality='transcriptomics'  # Anchor to static data
   )
   ```

3. **Weight the losses** (advanced):
   ```python
   # In custom training loop
   contrastive_loss_weighted = (
       2.0 * loss_transcriptomics +  # Higher weight for static
       1.0 * loss_proteomics
   ) / 3.0
   ```

---

## 4. Handling Dose-Response Relationships

### Question
"How to account for samples treated with different doses? Should I name them `sample1_dose1`, `sample1_dose2` or treat dose as metadata?"

### Recommended Approach: Dose as Metadata

**✅ RECOMMENDED**: Treat dose as a **numerical metadata feature**, not as separate samples.

```python
# metadata.csv structure
# sample_id,drug,cell_line,dose,response
# sample_001,Drug_A,HeLa,1.0,No_response
# sample_001,Drug_A,HeLa,5.0,Partial_response  # Same biological sample
# sample_001,Drug_A,HeLa,10.0,Full_response    # Same biological sample
# sample_002,Drug_B,MCF7,1.0,No_response       # Different sample
# sample_002,Drug_B,MCF7,5.0,Full_response

dataset = TemporalMultiOmicsDataset(
    static_data_paths={'transcriptomics': 'genes.csv'},
    metadata_path='metadata.csv',
    cat_cols=['drug', 'cell_line'],  # Categorical
    num_cols=['dose'],               # ✅ Dose as numerical feature
    label_col='response',
    normalize=True
)
```

### Why This Approach Works

1. **Model learns dose-response relationship**
   - The model can learn that higher dose → stronger response
   - Continuous relationship captured in embedding space

2. **Correct sample relationships preserved**
   - Samples from same cell line (sample_001) will have similar base embeddings
   - Dose modulates the prediction appropriately

3. **Enables dose interpolation**
   - Model can predict untested doses between 1.0 and 10.0

### Architecture Handles Dose

```python
# The model processes dose as numerical metadata
dose_embedding = metadata_encoder(
    cat_features=['Drug_A', 'HeLa'],   # Categorical
    num_features=[5.0]                  # Dose value
)

# Final embedding considers:
# 1. Molecular data (transcriptomics, proteomics)
# 2. Drug and cell line (categorical)
# 3. Dose (numerical) ← Dose-response captured here
```

### Alternative: Dose as Categorical

If doses are **discrete and few** (e.g., low/medium/high):

```python
# metadata.csv
# sample_id,drug,cell_line,dose_level,response
# sample_001,Drug_A,HeLa,low,No_response
# sample_001,Drug_A,HeLa,high,Full_response

dataset = TemporalMultiOmicsDataset(
    static_data_paths={'transcriptomics': 'genes.csv'},
    metadata_path='metadata.csv',
    cat_cols=['drug', 'cell_line', 'dose_level'],  # Dose as category
    label_col='response',
    normalize=True
)
```

### ❌ NOT RECOMMENDED: Separate Sample IDs per Dose

```python
# DON'T DO THIS:
# sample_id,drug,dose,response
# sample1_dose1,Drug_A,1.0,No_response
# sample1_dose2,Drug_A,5.0,Partial_response
# sample1_dose3,Drug_A,10.0,Full_response
```

**Problems:**
1. Model doesn't know these are related samples
2. Loses dose-response relationship
3. Treats as independent samples (incorrect)
4. Can't interpolate or extrapolate doses

### Special Case: Paired Sample Design

If you have **truly paired samples** (same cells, different treatments):

```python
# Option 1: Include pairing information
# metadata.csv
# sample_id,biological_replicate,drug,dose,response
# sample_001_treat1,rep_001,Drug_A,1.0,No_response
# sample_002_treat2,rep_001,Drug_A,5.0,Partial_response
# sample_003_treat3,rep_002,Drug_B,1.0,Full_response
# sample_004_treat4,rep_002,Drug_B,5.0,Full_response

dataset = TemporalMultiOmicsDataset(
    static_data_paths={'transcriptomics': 'genes.csv'},
    metadata_path='metadata.csv',
    cat_cols=['drug', 'biological_replicate'],  # Capture pairing
    num_cols=['dose'],
    label_col='response',
    normalize=True
)
```

### Analyzing Dose Effects

After training, analyze dose-response relationship:

```python
# Extract embeddings
embeddings, labels, predictions = evaluate_temporal_model(model, test_dataset, device)

# Load metadata to get doses
metadata = pd.read_csv('metadata.csv')
test_metadata = metadata.iloc[test_dataset.indices]

# Plot dose-response
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
for drug in test_metadata['drug'].unique():
    drug_mask = test_metadata['drug'] == drug
    doses = test_metadata[drug_mask]['dose']
    responses = predictions[drug_mask]
    
    ax.scatter(doses, responses, label=drug, alpha=0.6)

ax.set_xlabel('Dose (μM)')
ax.set_ylabel('Predicted Response')
ax.set_title('Dose-Response Relationship Learned by Model')
ax.legend()
plt.savefig('dose_response_curve.png', dpi=300, bbox_inches='tight')
```

### Summary Table

| Approach | Use When | Advantages | Disadvantages |
|----------|----------|------------|---------------|
| **Dose as numerical metadata** (RECOMMENDED) | Continuous doses (0.1, 1, 10, 100 μM) | Learns dose-response, can interpolate | Requires numerical encoding |
| **Dose as categorical metadata** | Few discrete levels (low, medium, high) | Simple encoding, interpretable | Can't interpolate, treats as unordered |
| **Dose in sample ID** | Never! | None | Breaks relationships, no dose-response learning |
| **Biological replicate + dose** | Paired experimental design | Captures pairing, more information | More complex, needs larger dataset |

---

## Complete Example: Putting It All Together

```python
import torch
import pandas as pd
import numpy as np
from torch.utils.data import random_split

from multiomicsbind import (
    TemporalMultiOmicsDataset,
    train_temporal_model,
    evaluate_temporal_model,
    create_analysis_report,
    check_and_fix_all_nan_values,
    compute_feature_importance
)

# 1. Create dataset with proper metadata
dataset = TemporalMultiOmicsDataset(
    static_data_paths={'transcriptomics': 'genes.csv'},
    temporal_data_paths={'proteomics': 'proteins_timeseries.csv'},
    temporal_metadata={'proteomics': {'timepoints': [0,1,2,4,8]}},
    metadata_path='metadata.csv',
    cat_cols=['drug', 'cell_line'],
    num_cols=['dose'],  # ✅ Dose as numerical feature
    label_col='response',  # Values: 0, 1, 2
    normalize=True
)

# 2. Fix NaN values
dataset, nan_summary = check_and_fix_all_nan_values(dataset, verbose=True)

# 3. Split data (no leakage!)
torch.manual_seed(42)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"\n✓ Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# Verify no leakage
train_indices = set(train_dataset.indices)
test_indices = set(test_dataset.indices)
assert len(train_indices & test_indices) == 0, "Data leakage!"
print("✓ No data leakage detected")

# 4. Train model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, history = train_temporal_model(
    dataset=train_dataset,  # ✅ Training data only
    device=device,
    binding_modality='transcriptomics',
    epochs=20,
    batch_size=32,
    verbose=True
)

# 5. Evaluate on test set
embeddings, labels, predictions = evaluate_temporal_model(
    model=model,
    dataset=test_dataset,  # ✅ Test data only
    device=device
)

test_accuracy = (predictions == labels).mean()
print(f"\n✓ Test Set Accuracy: {test_accuracy:.4f}")

# 6. Generate report with custom class names
class_names = ['No Response', 'Partial Response', 'Full Response']

report = create_analysis_report(
    model=model,
    dataset=test_dataset,  # ✅ Test set
    device=device,
    class_names=class_names,  # ✅ Custom labels
    history=history,
    output_dir='./results',
    compute_importance=True,
    compute_similarity=True,
    verbose=True
)

# 7. Analyze modality contributions
importance_dict, importance_df = compute_feature_importance(
    model=model,
    dataset=test_dataset,
    device=device,
    n_batches=10
)

# Check which modality contributes more
modality_contribution = importance_df.groupby('modality')['importance'].sum()
print("\nModality Contributions:")
print(modality_contribution)
print(f"\nTemporal proteomics contributes: {modality_contribution['proteomics'] / modality_contribution.sum() * 100:.1f}%")
print(f"Static transcriptomics contributes: {modality_contribution['transcriptomics'] / modality_contribution.sum() * 100:.1f}%")

# 8. Analyze dose-response relationship
metadata = pd.read_csv('metadata.csv')
test_metadata = metadata.iloc[test_dataset.indices]

print("\nDose-Response Analysis:")
for response_class in [0, 1, 2]:
    class_mask = labels == response_class
    mean_dose = test_metadata.iloc[class_mask]['dose'].mean()
    print(f"  {class_names[response_class]}: Mean dose = {mean_dose:.2f} μM")

print("\n✓ Analysis complete! Check ./results/ for all outputs:")
print("  - training_history.png (training curves)")
print("  - confusion_matrix.png (with custom class names)")
print("  - embeddings_umap_*.png (UMAP with custom class names)")
print("  - feature_importance.png (modality contributions)")
print("  - cross_modal_similarity.png (modality alignment)")
```

---

## Quick Reference

### UMAP with Class Names
```python
report = create_analysis_report(..., class_names=['Class_A', 'Class_B', 'Class_C'])
```

### No Data Leakage
```python
train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])
model, history = train_temporal_model(dataset=train_dataset, ...)
report = create_analysis_report(model=model, dataset=test_dataset, ...)
```

### Temporal vs Static Contribution
```python
importance_df.groupby('modality')['importance'].sum()
```

### Dose as Metadata
```python
dataset = TemporalMultiOmicsDataset(..., num_cols=['dose'], ...)
```

---

## Additional Resources

- See `BEST_PRACTICES.md` for general guidelines
- See `examples/temporal_example.py` for complete working example
- See API documentation for detailed parameter descriptions
