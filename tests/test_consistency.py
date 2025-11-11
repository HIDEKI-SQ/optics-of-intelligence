"""E8 Series - E8e Consistency Tests

Tests for consistency with E8e theoretical predictions (O1-O4).

References:
    E8b (2025-138): Empirical validation of four orthogonality laws
    E8c (2025-139): Theoretical framework
    E8d (2025-140): Memory palace reinterpretation

Author: HIDEKI
Date: 2025-11
License: MIT
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import pytest


def compute_vs(D_semantic, D_spatial):
    """Compute VS (Value-Space correlation).
    
    Args:
        D_semantic: Semantic distance matrix
        D_spatial: Spatial distance matrix
        
    Returns:
        Spearman correlation coefficient
    """
    sem_flat = squareform(D_semantic, checks=False)
    spatial_flat = squareform(D_spatial, checks=False)
    vs, _ = spearmanr(sem_flat, spatial_flat)
    return vs


class TestO1NaturalOrthogonality:
    """Test O1: Natural Orthogonality (VS≈0 without value pressure)."""
    
    def test_random_baseline(self):
        """Test VS≈0 for completely random arrangements."""
        np.random.seed(42)
        n_items = 20
        dim = 100
        n_trials = 100
        
        vs_values = []
        for _ in range(n_trials):
            # Random semantic
            A = np.random.randn(n_items, dim)
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            D_semantic = squareform(pdist(A, metric='correlation'))
            
            # Random spatial
            coords = np.random.randn(n_items, 2)
            D_spatial = squareform(pdist(coords, metric='euclidean'))
            
            vs = compute_vs(D_semantic, D_spatial)
            vs_values.append(vs)
        
        vs_mean = np.mean(vs_values)
        
        # O1 prediction: VS≈0
        assert abs(vs_mean) < 0.1, \
            f"O1 violated: Expected VS≈0, got {vs_mean:.4f}"
    
    def test_dimension_invariance(self):
        """Test O1 holds across embedding dimensions."""
        np.random.seed(42)
        n_items = 20
        dimensions = [10, 50, 100, 200]
        
        for dim in dimensions:
            A = np.random.randn(n_items, dim)
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            D_semantic = squareform(pdist(A, metric='correlation'))
            
            coords = np.random.randn(n_items, 2)
            D_spatial = squareform(pdist(coords, metric='euclidean'))
            
            vs = compute_vs(D_semantic, D_spatial)
            
            # O1 should hold regardless of dimension
            assert abs(vs) < 0.2, \
                f"O1 violated at dim={dim}: VS={vs:.4f}"
    
    def test_spatial_structure_invariance(self):
        """Test O1 holds for different spatial structures."""
        np.random.seed(42)
        n_items = 20
        dim = 100
        
        # Fixed semantic structure
        A = np.random.randn(n_items, dim)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        D_semantic = squareform(pdist(A, metric='correlation'))
        
        # Test different spatial arrangements
        spatial_types = ['random', 'grid', 'circle']
        
        for spatial_type in spatial_types:
            if spatial_type == 'random':
                coords = np.random.randn(n_items, 2)
            elif spatial_type == 'grid':
                grid_size = int(np.ceil(np.sqrt(n_items)))
                x = np.arange(n_items) % grid_size
                y = np.arange(n_items) // grid_size
                coords = np.column_stack([x, y])
            elif spatial_type == 'circle':
                angles = 2 * np.pi * np.arange(n_items) / n_items
                coords = np.column_stack([np.cos(angles), np.sin(angles)])
            
            D_spatial = squareform(pdist(coords, metric='euclidean'))
            vs = compute_vs(D_semantic, D_spatial)
            
            # O1 should hold for all spatial structures
            assert abs(vs) < 0.2, \
                f"O1 violated for {spatial_type}: VS={vs:.4f}"


class TestO2PhaseDominance:
    """Test O2: Phase Dominance (topology matters, not absolute position)."""
    
    def test_rotation_invariance(self):
        """Test VS≈0 is preserved under rotation."""
        np.random.seed(42)
        n_items = 20
        dim = 100
        
        # Fixed semantic structure
        A = np.random.randn(n_items, dim)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        D_semantic = squareform(pdist(A, metric='correlation'))
        
        # Circle arrangement
        angles = 2 * np.pi * np.arange(n_items) / n_items
        coords = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Test rotations
        rotation_angles = [0, 45, 90, 180]
        vs_values = []
        
        for angle_deg in rotation_angles:
            angle_rad = np.deg2rad(angle_deg)
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            
            rotated_coords = coords @ rotation_matrix.T
            D_spatial = squareform(pdist(rotated_coords, metric='euclidean'))
            vs = compute_vs(D_semantic, D_spatial)
            vs_values.append(vs)
        
        vs_std = np.std(vs_values)
        
        # O2 prediction: VS stable under rotation
        assert vs_std < 0.1, \
            f"O2 violated: VS should be rotation-invariant, got std={vs_std:.4f}"
    
    def test_topological_disruption_increases_vs(self):
        """Test that breaking topology increases VS."""
        np.random.seed(42)
        n_items = 20
        dim = 100
        
        # Semantic structure with local similarity
        A = np.zeros((n_items, dim))
        for i in range(n_items):
            A[i] = np.random.randn(dim) + i * 0.1  # Progressive shift
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        D_semantic = squareform(pdist(A, metric='correlation'))
        
        # Ordered circle (preserves topology)
        angles = 2 * np.pi * np.arange(n_items) / n_items
        coords_ordered = np.column_stack([np.cos(angles), np.sin(angles)])
        D_spatial_ordered = squareform(pdist(coords_ordered, metric='euclidean'))
        vs_ordered = compute_vs(D_semantic, D_spatial_ordered)
        
        # Shuffled circle (breaks topology)
        perm = np.random.permutation(n_items)
        coords_shuffled = coords_ordered[perm]
        D_spatial_shuffled = squareform(pdist(coords_shuffled, metric='euclidean'))
        vs_shuffled = compute_vs(D_semantic, D_spatial_shuffled)
        
        # O2 prediction: Breaking topology increases |VS|
        assert abs(vs_shuffled) > abs(vs_ordered), \
            f"O2 violated: Topology break should increase |VS|"


class TestO3StressTolerance:
    """Test O3: Stress Tolerance (VS≈0 robust to perturbations)."""
    
    def test_coordinate_noise_tolerance(self):
        """Test VS≈0 maintained under coordinate noise."""
        np.random.seed(42)
        n_items = 20
        dim = 100
        
        # Random semantic
        A = np.random.randn(n_items, dim)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        D_semantic = squareform(pdist(A, metric='correlation'))
        
        # Circle with increasing noise
        angles = 2 * np.pi * np.arange(n_items) / n_items
        coords = np.column_stack([np.cos(angles), np.sin(angles)])
        
        noise_levels = [0.0, 0.1, 0.2, 0.5]
        vs_values = []
        
        for noise in noise_levels:
            coords_noisy = coords + np.random.randn(n_items, 2) * noise
            D_spatial = squareform(pdist(coords_noisy, metric='euclidean'))
            vs = compute_vs(D_semantic, D_spatial)
            vs_values.append(vs)
        
        # O3 prediction: VS remains ≈0 despite noise
        for noise, vs in zip(noise_levels, vs_values):
            assert abs(vs) < 0.3, \
                f"O3 violated at noise={noise}: VS={vs:.4f}"
    
    def test_semantic_noise_tolerance(self):
        """Test VS≈0 maintained under semantic noise."""
        np.random.seed(42)
        n_items = 20
        dim = 100
        
        # Base semantic structure
        A_base = np.random.randn(n_items, dim)
        A_base = A_base / (np.linalg.norm(A_base, axis=1, keepdims=True) + 1e-12)
        
        # Fixed spatial
        angles = 2 * np.pi * np.arange(n_items) / n_items
        coords = np.column_stack([np.cos(angles), np.sin(angles)])
        D_spatial = squareform(pdist(coords, metric='euclidean'))
        
        # Add semantic noise
        noise_levels = [0.0, 0.2, 0.5, 1.0]
        vs_values = []
        
        for noise in noise_levels:
            A_random = np.random.randn(n_items, dim)
            A_random = A_random / (np.linalg.norm(A_random, axis=1, keepdims=True) + 1e-12)
            
            A_noisy = (1 - noise) * A_base + noise * A_random
            A_noisy = A_noisy / (np.linalg.norm(A_noisy, axis=1, keepdims=True) + 1e-12)
            
            D_semantic = squareform(pdist(A_noisy, metric='correlation'))
            vs = compute_vs(D_semantic, D_spatial)
            vs_values.append(vs)
        
        # O3 prediction: VS remains ≈0 despite semantic noise
        for noise, vs in zip(noise_levels, vs_values):
            assert abs(vs) < 0.3, \
                f"O3 violated at semantic noise={noise}: VS={vs:.4f}"


class TestO4ValueGatedCoupling:
    """Test O4: Value-Gated Coupling (λ controls VS)."""
    
    def test_lambda_monotonic_increase(self):
        """Test VS increases monotonically with λ."""
        np.random.seed(42)
        n_items = 20
        dim = 100
        
        # Semantic structure
        A = np.random.randn(n_items, dim)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        D_semantic = squareform(pdist(A, metric='correlation'))
        
        lambda_values = [0.0, 0.2, 0.5, 0.8, 1.0]
        vs_values = []
        
        for lam in lambda_values:
            # Interpolate between random and semantic-aligned spatial
            D_random = np.random.rand(n_items, n_items)
            D_random = (D_random + D_random.T) / 2
            np.fill_diagonal(D_random, 0)
            
            D_spatial = (1 - lam) * D_random + lam * D_semantic
            vs = compute_vs(D_semantic, D_spatial)
            vs_values.append(vs)
        
        # O4 prediction: VS increases with λ
        for i in range(len(vs_values) - 1):
            assert vs_values[i+1] > vs_values[i] - 0.1, \
                f"O4 violated: VS should increase with λ"
    
    def test_lambda_extremes(self):
        """Test VS at λ=0 (random) and λ=1 (aligned)."""
        np.random.seed(42)
        n_items = 20
        dim = 100
        
        A = np.random.randn(n_items, dim)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        D_semantic = squareform(pdist(A, metric='correlation'))
        
        # λ=0: Random spatial
        D_random = np.random.rand(n_items, n_items)
        D_random = (D_random + D_random.T) / 2
        np.fill_diagonal(D_random, 0)
        vs_random = compute_vs(D_semantic, D_random)
        
        # λ=1: Perfectly aligned
        vs_aligned = compute_vs(D_semantic, D_semantic)
        
        # O4 predictions
        assert abs(vs_random) < 0.3, \
            f"O4 violated at λ=0: Expected VS≈0, got {vs_random:.4f}"
        assert vs_aligned > 0.9, \
            f"O4 violated at λ=1: Expected VS≈1, got {vs_aligned:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
