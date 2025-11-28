"""
Test script to verify reproducibility with set_seed function.
This demonstrates that MultiOmicsBind produces identical results across multiple runs.
"""

import numpy as np
import torch
from multiomicsbind import set_seed

print("=" * 80)
print("Testing MultiOmicsBind Reproducibility")
print("=" * 80)

# Test 1: Verify seed setting works
print("\n📌 Test 1: Random number generation reproducibility")
print("-" * 80)

# Run 1
set_seed(42, verbose=False)
numpy_random_1 = np.random.rand(5)
torch_random_1 = torch.rand(5)

# Run 2 (same seed)
set_seed(42, verbose=False)
numpy_random_2 = np.random.rand(5)
torch_random_2 = torch.rand(5)

# Check if identical
numpy_match = np.allclose(numpy_random_1, numpy_random_2)
torch_match = torch.allclose(torch_random_1, torch_random_2)

print(f"NumPy random (Run 1): {numpy_random_1}")
print(f"NumPy random (Run 2): {numpy_random_2}")
print(f"✓ NumPy reproducible: {numpy_match}")

print(f"\nPyTorch random (Run 1): {torch_random_1.numpy()}")
print(f"PyTorch random (Run 2): {torch_random_2.numpy()}")
print(f"✓ PyTorch reproducible: {torch_match}")

# Test 2: Neural network initialization reproducibility
print("\n\n📌 Test 2: Neural network weight initialization reproducibility")
print("-" * 80)

# Run 1
set_seed(42, verbose=False)
model_1 = torch.nn.Linear(10, 5)
weights_1 = model_1.weight.data.clone()

# Run 2 (same seed)
set_seed(42, verbose=False)
model_2 = torch.nn.Linear(10, 5)
weights_2 = model_2.weight.data.clone()

# Check if identical
weights_match = torch.allclose(weights_1, weights_2)
print(f"Model 1 weights (first 5): {weights_1[0, :5]}")
print(f"Model 2 weights (first 5): {weights_2[0, :5]}")
print(f"✓ Model initialization reproducible: {weights_match}")

# Test 3: Different seed produces different results
print("\n\n📌 Test 3: Different seeds produce different results")
print("-" * 80)

set_seed(42, verbose=False)
random_seed_42 = torch.rand(5)

set_seed(123, verbose=False)
random_seed_123 = torch.rand(5)

different = not torch.allclose(random_seed_42, random_seed_123)
print(f"Random with seed=42:  {random_seed_42.numpy()}")
print(f"Random with seed=123: {random_seed_123.numpy()}")
print(f"✓ Different seeds produce different results: {different}")

# Summary
print("\n" + "=" * 80)
print("✅ REPRODUCIBILITY TEST RESULTS")
print("=" * 80)
all_pass = numpy_match and torch_match and weights_match and different

if all_pass:
    print("🎉 All tests passed!")
    print("\n💡 Usage in your code:")
    print("   from multiomicsbind import set_seed")
    print("   ")
    print("   # Set seed at the beginning of your script")
    print("   set_seed(42)")
    print("   ")
    print("   # Now all operations are reproducible")
    print("   model = MultiOmicsBindWithHead(...)")
    print("   dataset = MultiOmicsDataset(...)")
    print("   train_multiomicsbind(model, ...)")
else:
    print("❌ Some tests failed")
    print(f"   NumPy: {'✓' if numpy_match else '✗'}")
    print(f"   PyTorch: {'✓' if torch_match else '✗'}")
    print(f"   Model init: {'✓' if weights_match else '✗'}")
    print(f"   Different seeds: {'✓' if different else '✗'}")

print("=" * 80)
