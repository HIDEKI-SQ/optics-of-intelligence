"""E8 Series - Basic Functionality Tests

Tests for core functionality and data structure integrity.

Author: HIDEKI
Date: 2025-11
License: MIT
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import pytest


def test_vs_computation():
    """Test VS (Value-Space) computation is correct.
    
    Verifies that VS correlation is computed properly using Spearman's rho.
    """
    # Create simple test case
    D_semantic = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0]
    ])
    
    D_spatial = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0]
    ])
    
    # Compute VS
    sem_flat = squareform(D_semantic, checks=False)
    spatial_flat = squareform(D_spatial, checks=False)
    vs, _ = spearmanr(sem_flat, spatial_flat)
    
    # Perfect correlation expected
    assert abs(vs - 1.0) < 1e-10, f"Expected VS=1.0, got {vs}"


def test_vs_random_independence():
    """Test VS≈0 for independent random structures.
    
    Verifies O1: Natural Orthogonality for random semantic and spatial structures.
    """
    np.random.seed(42)
    n_items = 20
    dim = 100
    n_trials = 100
    
    vs_values = []
    
    for _ in range(n_trials):
        # Random semantic structure
        A = np.random.randn(n_items, dim)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        D_semantic = squareform(pdist(A, metric='correlation'))
        
        # Random spatial structure
        coords = np.random.randn(n_items, 2)
        D_spatial = squareform(pdist(coords, metric='euclidean'))
        
        # Compute VS
        sem_flat = squareform(D_semantic, checks=False)
        spatial_flat = squareform(D_spatial, checks=False)
        vs, _ = spearmanr(sem_flat, spatial_flat)
        vs_values.append(vs)
    
    vs_mean = np.mean(vs_values)
    vs_std = np.std(vs_values)
    
    # VS should be close to 0
    assert abs(vs_mean) < 0.1, f"Expected VS≈0, got {vs_mean:.4f}"
    assert vs_std < 0.15, f"Expected stable VS, got std={vs_std:.4f}"


def test_distance_matrix_symmetry():
    """Test that distance matrices are symmetric."""
    np.random.seed(42)
    n_items = 10
    
    # Random points
    points = np.random.randn(n_items, 3)
    D = squareform(pdist(points))
    
    # Check symmetry
    assert np.allclose(D, D.T), "Distance matrix should be symmetric"
    
    # Check diagonal is zero
    assert np.allclose(np.diag(D), 0), "Distance matrix diagonal should be zero"


def test_correlation_distance_range():
    """Test that correlation distance is in [0, 2]."""
    np.random.seed(42)
    n_items = 20
    dim = 50
    
    A = np.random.randn(n_items, dim)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    
    D = squareform(pdist(A, metric='correlation'))
    
    # Correlation distance range
    assert np.all(D >= 0), "Correlation distance should be non-negative"
    assert np.all(D <= 2), "Correlation distance should be <= 2"


def test_embedding_normalization():
    """Test that embeddings are properly normalized."""
    np.random.seed(42)
    n_items = 10
    dim = 100
    
    # Random embeddings
    A = np.random.randn(n_items, dim)
    
    # Normalize
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    A_normalized = A / (norms + 1e-12)
    
    # Check unit norm
    computed_norms = np.linalg.norm(A_normalized, axis=1)
    expected_norms = np.ones(n_items)
    
    assert np.allclose(computed_norms, expected_norms, atol=1e-10), \
        "Normalized embeddings should have unit norm"


def test_spearman_correlation_range():
    """Test that Spearman correlation is in [-1, 1]."""
    np.random.seed(42)
    n_trials = 50
    
    for _ in range(n_trials):
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        rho, _ = spearmanr(x, y)
        
        assert -1 <= rho <= 1, f"Spearman correlation should be in [-1, 1], got {rho}"


def test_reproducibility():
    """Test that results are reproducible with fixed seed."""
    n_items = 15
    dim = 50
    
    # First run
    np.random.seed(123)
    A1 = np.random.randn(n_items, dim)
    A1 = A1 / (np.linalg.norm(A1, axis=1, keepdims=True) + 1e-12)
    D1 = squareform(pdist(A1, metric='correlation'))
    
    # Second run with same seed
    np.random.seed(123)
    A2 = np.random.randn(n_items, dim)
    A2 = A2 / (np.linalg.norm(A2, axis=1, keepdims=True) + 1e-12)
    D2 = squareform(pdist(A2, metric='correlation'))
    
    # Should be identical
    assert np.allclose(D1, D2), "Results should be reproducible with fixed seed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
