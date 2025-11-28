"""
Test script to verify feature importance normalization.
"""

import numpy as np
import pandas as pd

# Simulate what happens in compute_feature_importance
print("=" * 80)
print("Testing Feature Importance Normalization")
print("=" * 80)

# Simulate raw gradient scores for different modalities
raw_scores = {
    'mRNA': np.array([0.0001, 0.0025, 0.0003, 0.0050, 0.0002]),  # Small scale
    'proteomics': np.array([0.15, 0.82, 0.23, 0.91, 0.05]),      # Medium scale
    'methylation': np.array([5.2, 15.8, 8.3, 22.1, 3.7])        # Large scale
}

print("\n📊 Raw Gradient Scores (before normalization):")
print("-" * 80)
for modality, scores in raw_scores.items():
    print(f"{modality:15s}: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

# Apply normalization (same as in the updated code)
normalized_scores = {}
for modality, scores in raw_scores.items():
    min_score = scores.min()
    max_score = scores.max()
    if max_score > min_score:
        normalized = (scores - min_score) / (max_score - min_score)
    else:
        normalized = np.zeros_like(scores)
    normalized_scores[modality] = normalized

print("\n✨ Normalized Scores (after normalization to [0, 1]):")
print("-" * 80)
for modality, scores in normalized_scores.items():
    print(f"{modality:15s}: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    print(f"{'':15s}  values: {scores}")

print("\n" + "=" * 80)
print("✅ Verification:")
print("=" * 80)
all_valid = True
for modality, scores in normalized_scores.items():
    min_ok = np.isclose(scores.min(), 0.0)
    max_ok = np.isclose(scores.max(), 1.0)
    range_ok = np.all((scores >= 0) & (scores <= 1))
    
    status = "✅" if (min_ok and max_ok and range_ok) else "❌"
    print(f"{status} {modality:15s}: min≈0? {min_ok}, max≈1? {max_ok}, all in [0,1]? {range_ok}")
    all_valid = all_valid and min_ok and max_ok and range_ok

if all_valid:
    print("\n🎉 All modalities correctly normalized to [0, 1] range!")
else:
    print("\n❌ Some modalities failed normalization check")

print("\n💡 Key Benefits:")
print("   • Scores are now comparable across modalities")
print("   • Easy to interpret: 1.0 = most important, 0.0 = least important")
print("   • Consistent scale regardless of gradient magnitudes")
print("=" * 80)
