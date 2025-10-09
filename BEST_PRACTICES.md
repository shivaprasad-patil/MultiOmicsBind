# Best Practices for MultiOmicsBind

## Table of Contents
- [Avoiding Data Leakage](#avoiding-data-leakage)
- [Train/Validation/Test Splits](#trainvalidationtest-splits)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Model Evaluation](#model-evaluation)
- [Feature Importance Analysis](#feature-importance-analysis)
- [Cross-Modal Similarity](#cross-modal-similarity)
- [Hyperparameter Tuning](#hyperparameter-tuning)

---

## Avoiding Data Leakage

### What is Data Leakage?

**Data leakage** occurs when information from the test set "leaks" into the training process, leading to overly optimistic performance estimates that don't generalize to new data.

### Common Data Leakage Mistakes

#### ❌ WRONG: Evaluating on Training Data
```python
# Train on full dataset
model, history = train_temporal_model(dataset, device, epochs=20)

# Evaluate on SAME dataset - this is DATA LEAKAGE!
embeddings, labels, predictions = evaluate_temporal_model(model, dataset, device)
accuracy = (predictions == labels).mean()
print(f"Accuracy: {accuracy:.4f}")  # ❌ Misleading! Will be too high
```

**Why is this wrong?**
- The model has already seen and memorized these samples
- Accuracy will be artificially high (often 100%)
- Performance won't generalize to new, unseen data

#### ✅ CORRECT: Proper Train/Test Split
```python
from torch.utils.data import random_split

# Split data BEFORE training
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Train on training set only
model, history = train_temporal_model(train_dataset, device, epochs=20)

# Evaluate on held-out test set
embeddings, labels, predictions = evaluate_temporal_model(model, test_dataset, device)
accuracy = (predictions == labels).mean()
print(f"Test Accuracy: {accuracy:.4f}")  # ✅ Realistic estimate!
```

---

## Train/Validation/Test Splits

### Recommended Split Ratios

For most biological datasets:
- **Training**: 60-70% (for learning patterns)
- **Validation**: 15-20% (for hyperparameter tuning)
- **Test**: 15-20% (for final evaluation)

### Three-Way Split Example

```python
from torch.utils.data import random_split

# Calculate sizes
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Create splits
torch.manual_seed(42)  # For reproducibility
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

print(f"Training samples: {len(train_dataset)} ({100*train_size/total_size:.0f}%)")
print(f"Validation samples: {len(val_dataset)} ({100*val_size/total_size:.0f}%)")
print(f"Test samples: {len(test_dataset)} ({100*test_size/total_size:.0f}%)")

# Train with validation
model, history = train_temporal_model(
    dataset=train_dataset,
    device=device,
    val_split=0.0,  # No additional split since we already have val_dataset
    epochs=20
)

# Tune hyperparameters using validation set
val_embeddings, val_labels, val_predictions = evaluate_temporal_model(
    model, val_dataset, device
)
val_accuracy = (val_predictions == val_labels).mean()

# Final evaluation on test set (only once!)
test_embeddings, test_labels, test_predictions = evaluate_temporal_model(
    model, test_dataset, device
)
test_accuracy = (test_predictions == test_labels).mean()

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```

### When to Use Which Set?

| Dataset | Purpose | Frequency of Use |
|---------|---------|-----------------|
| **Training** | Learn model parameters | Every training epoch |
| **Validation** | Tune hyperparameters, early stopping | During hyperparameter search |
| **Test** | Report final performance | **ONCE** at the very end |

⚠️ **CRITICAL**: Never touch the test set until you've finalized your model!

---

## Synthetic Data Generation

### Creating Realistic Synthetic Data

When generating synthetic data for testing or examples, make it **challenging but realistic**:

#### ❌ WRONG: Too Easy/Deterministic
```python
# This will lead to 100% accuracy (unrealistic)
for i in range(n_samples):
    if labels[i] == 0:
        features[:100] += np.random.normal(-2.0, 0.3, 100)  # Strong, clean signal
    elif labels[i] == 1:
        features[:100] += np.random.normal(0.0, 0.3, 100)
    else:
        features[:100] += np.random.normal(2.0, 0.3, 100)   # Perfectly separated
```

**Problems:**
- Signal-to-noise ratio too high (6.7:1)
- Classes perfectly separated
- Too many predictive features (100)
- No missing data
- Every sample has identical pattern

#### ✅ CORRECT: Realistic and Challenging
```python
# Realistic biological data: noisy, weak signals, missing values
for i in range(n_samples):
    # Variable signal strength per sample (biological variability)
    signal_strength = np.random.uniform(0.3, 0.7)
    noise_level = np.random.uniform(1.5, 2.5)
    
    # Weak, noisy signals with class overlap
    if labels[i] == 0:
        # Only 30 genes, weak downregulation
        features[:30] += np.random.normal(-0.3 * signal_strength, noise_level, 30)
    elif labels[i] == 1:
        # 50 genes, moderate upregulation
        features[:50] += np.random.normal(0.4 * signal_strength, noise_level, 50)
    else:
        # 80 genes, stronger upregulation (but still noisy)
        features[:80] += np.random.normal(0.8 * signal_strength, noise_level, 80)
    
    # Add realistic missing data (2-5%)
    dropout_mask = np.random.random(n_features) < 0.03
    features[dropout_mask] = np.nan
```

**Better because:**
- Signal-to-noise ratio realistic (0.2-0.4:1)
- Classes overlap in feature space
- Fewer predictive features (30-80)
- Variable signal strength (biological variability)
- Missing data simulates real-world measurements
- Expected accuracy: **60-85%** (realistic for biology)

### Realistic Data Characteristics

Real biological data typically has:
- **High noise**: Technical + biological variability
- **Weak signals**: Most biomarkers have subtle effects
- **Class overlap**: Different phenotypes aren't perfectly separated
- **Missing values**: Dropout, technical failures, batch effects
- **Batch effects**: Systematic non-biological variation
- **Sample heterogeneity**: Individual variability

---

## Model Evaluation

### Comprehensive Evaluation Metrics

Don't rely on accuracy alone! Use multiple metrics:

```python
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# Get predictions on test set
embeddings, labels, predictions = evaluate_temporal_model(
    model, test_dataset, device
)

# 1. Overall accuracy
accuracy = (predictions == labels).mean()
print(f"Accuracy: {accuracy:.4f}")

# 2. Per-class metrics (precision, recall, F1)
print("\nClassification Report:")
print(classification_report(
    labels, predictions,
    target_names=['No Response', 'Partial', 'Full Response']
))

# 3. Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(labels, predictions))

# 4. For binary classification: ROC-AUC, PR-AUC
if len(np.unique(labels)) == 2:
    # Get predicted probabilities
    model.eval()
    with torch.no_grad():
        logits, _ = model(test_data)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
```

### Interpreting Results

**Good model performance on real biological data:**
- Accuracy: 65-85%
- Consistent across classes (check confusion matrix)
- Validation ≈ Test accuracy (no overfitting)
- Improvement over baseline/random

**Red flags:**
- 🚩 Accuracy = 100% → Data leakage or data too easy
- 🚩 Train accuracy >> Test accuracy → Overfitting
- 🚩 One class performs much worse → Class imbalance issues
- 🚩 Test accuracy < 60% → Insufficient signal, need more data/features

---

## Feature Importance Analysis

### Computing Feature Importance on Test Data

```python
# Compute importance using the test set (not training set!)
importance_dict, importance_df = compute_feature_importance(
    model=model,
    dataset=test_dataset,  # Use test set
    device=device,
    n_batches=10,
    batch_size=16
)

# Analyze top features
print("\nTop 20 Most Important Features:")
print(importance_df.head(20))

# Check feature importance distribution by modality
for modality in importance_df['modality'].unique():
    modality_features = importance_df[importance_df['modality'] == modality]
    print(f"\n{modality}:")
    print(f"  Total features: {len(modality_features)}")
    print(f"  Mean importance: {modality_features['importance'].mean():.6f}")
    print(f"  Top feature: {modality_features.iloc[0]['feature_name']}")
```

### Validating Feature Importance

To ensure feature importance is reliable:

1. **Consistency across runs**: Run multiple times with different random seeds
2. **Biological plausibility**: Do top features make sense biologically?
3. **Stability**: Top features should be stable across cross-validation folds
4. **Validation**: Test on independent validation set

---

## Cross-Modal Similarity

### Analyzing Cross-Modal Alignment

```python
# Compute similarity on test set
similarity_matrices = compute_cross_modal_similarity(
    embeddings_dict=test_embeddings,
    verbose=True
)

# Analyze per-class similarity
for comparison, sim_matrix in similarity_matrices.items():
    print(f"\n{comparison}:")
    
    # Overall similarity
    mean_sim = np.mean(sim_matrix)
    diagonal_sim = np.mean(np.diag(sim_matrix))
    
    print(f"  Overall mean: {mean_sim:.4f}")
    print(f"  Diagonal (same sample): {diagonal_sim:.4f}")
    
    # Per-class similarity
    for class_label in np.unique(labels):
        class_mask = labels == class_label
        class_sim = sim_matrix[class_mask][:, class_mask]
        print(f"  Class {class_label} mean: {np.mean(class_sim):.4f}")
```

### Expected Similarity Patterns

**Good cross-modal alignment:**
- Diagonal (same sample) > Off-diagonal (different samples)
- Similar patterns across all modalities
- Class-specific patterns visible

**Poor alignment:**
- Diagonal ≈ Off-diagonal → Modalities not aligned
- One modality very different from others → Check data quality
- No class structure → Insufficient signal

---

## Hyperparameter Tuning

### Using Validation Set for Tuning

```python
from itertools import product

# Define hyperparameter grid
embed_dims = [128, 256, 512]
learning_rates = [1e-4, 1e-3, 1e-2]
dropout_rates = [0.1, 0.2, 0.3]

best_val_acc = 0
best_params = None

# Grid search using VALIDATION set
for embed_dim, lr, dropout in product(embed_dims, learning_rates, dropout_rates):
    print(f"\nTrying: embed_dim={embed_dim}, lr={lr}, dropout={dropout}")
    
    # Train on training set
    model, history = train_temporal_model(
        dataset=train_dataset,
        device=device,
        embed_dim=embed_dim,
        lr=lr,
        dropout=dropout,
        epochs=20,
        val_split=0.0  # We have separate val_dataset
    )
    
    # Evaluate on VALIDATION set (NOT test!)
    _, val_labels, val_preds = evaluate_temporal_model(
        model, val_dataset, device
    )
    val_acc = (val_preds == val_labels).mean()
    
    print(f"  Validation accuracy: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = (embed_dim, lr, dropout)

print(f"\nBest parameters: embed_dim={best_params[0]}, lr={best_params[1]}, dropout={best_params[2]}")
print(f"Best validation accuracy: {best_val_acc:.4f}")

# NOW train final model with best params and evaluate on TEST set
final_model, final_history = train_temporal_model(
    dataset=train_dataset,
    device=device,
    embed_dim=best_params[0],
    lr=best_params[1],
    dropout=best_params[2],
    epochs=20
)

# FINAL evaluation on test set (report this!)
_, test_labels, test_preds = evaluate_temporal_model(
    final_model, test_dataset, device
)
test_acc = (test_preds == test_labels).mean()
print(f"\nFinal TEST accuracy: {test_acc:.4f}")
```

---

## Summary Checklist

Before publishing results, verify:

- [ ] ✅ Proper train/validation/test split implemented
- [ ] ✅ Test set never used during training or hyperparameter tuning
- [ ] ✅ All evaluation functions receive test/validation sets, not training data
- [ ] ✅ Synthetic data (if used) has realistic signal-to-noise ratio
- [ ] ✅ Multiple evaluation metrics computed (not just accuracy)
- [ ] ✅ Cross-validation performed for robustness (optional but recommended)
- [ ] ✅ Feature importance validated across multiple runs
- [ ] ✅ Results are biologically interpretable
- [ ] ✅ Performance is realistic for biological data (not 100%)
- [ ] ✅ All random seeds set for reproducibility

---

## References

- **Avoiding Data Leakage**: [Preventing Data Leakage in Machine Learning](https://machinelearningmastery.com/data-leakage-machine-learning/)
- **Cross-Validation**: [Cross-validation for Biological Data](https://www.nature.com/articles/s41596-021-00543-7)
- **Model Evaluation**: [Best Practices in ML for Healthcare](https://www.nature.com/articles/s41591-023-02445-z)
- **Reproducibility**: [Guidelines for Reproducible ML Research](https://arxiv.org/abs/2003.12206)

---

## Questions?

If you're unsure whether your evaluation setup avoids data leakage:

1. Ask yourself: "Has the model seen this data during training?"
2. If yes → Data leakage! Use a different split.
3. If no → You're good!

For more help, see the [examples](examples/) directory for complete working examples with proper train/test splits.
