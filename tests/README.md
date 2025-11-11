# E8 Series Tests

## Overview

This directory contains tests for the E8 experimental series, ensuring:
1. Basic functionality of core components
2. Consistency with E8e theoretical predictions
3. Reproducibility of results

## Test Files

### `test_basic.py`
Basic functionality tests:
- VS computation correctness
- Distance matrix generation
- Data structure integrity

### `test_consistency.py`
E8e consistency tests:
- O1: Natural Orthogonality (VS≈0 for random arrangements)
- O2: Phase Dominance (topology preservation)
- O3: Stress Tolerance (robustness to noise)
- O4: Value-Gated Coupling (λ parameter effects)

## Running Tests

### Run all tests
```bash
python -m pytest tests/
```

### Run specific test file
```bash
python -m pytest tests/test_basic.py
python -m pytest tests/test_consistency.py
```

### Run with verbose output
```bash
python -m pytest tests/ -v
```

## Test Requirements

Tests require:
- numpy
- scipy
- pandas
- pytest (install: `pip install pytest`)

## Expected Results

All tests should pass, confirming:
- ✅ Core algorithms work correctly
- ✅ Results align with E8e predictions
- ✅ Statistical properties are maintained
