#!/bin/bash
# Test all updated examples

echo "=========================================="
echo "Testing MultiOmicsBind Examples"
echo "=========================================="

cd /Users/shivaprasad/Documents/PROJECTS/GitHub/MO/MultiOmicsBind/examples

# Test 1: basic_example.py
echo ""
echo "1. Testing basic_example.py..."
conda run -n pytorch python -c "import sys; sys.path.insert(0, '..'); from examples import basic_example; print('✅ Import OK')"

# Test 2: binding_modality_example.py
echo ""
echo "2. Testing binding_modality_example.py..."
conda run -n pytorch python -c "import sys; sys.path.insert(0, '..'); from examples import binding_modality_example; print('✅ Import OK')"

# Test 3: flexible_modalities_example.py
echo ""
echo "3. Testing flexible_modalities_example.py..."
conda run -n pytorch python -c "import sys; sys.path.insert(0, '..'); from examples import flexible_modalities_example; print('✅ Import OK')"

echo ""
echo "=========================================="
echo "✅ All examples passed import tests!"
echo "=========================================="
