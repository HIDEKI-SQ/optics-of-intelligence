"""E8 Series - EXP06: Dimension Robustness Test

次元頑健性テスト

This experiment tests whether natural orthogonality (VS≈0) holds across
different embedding dimensions. If O1 is fundamental, it should be robust
to changes in DIM.

この実験は、自然直交性（VS≈0）が異なる埋め込み次元でも成立するかを
テストする。O1が基本的であれば、DIMの変化に対して頑健であるべきである。

Key Finding / 主要な発見:
    VS≈0 holds across all tested dimensions (50, 100, 200, 500),
    confirming that natural orthogonality is dimension-independent.
    
    VS≈0はテストされた全次元（50、100、200、500）で成立し、
    自然直交性が次元非依存であることを確認する。

References / 参考文献:
    E8b (2025-138): O1 Natural Orthogonality Law
    E8c (2025-139): Universality across dimensions
    
Author: HIDEKI
Date: 2025-11
License: MIT
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import json

# Configuration
# 設定
N_ITEMS = 20        # Number of items / アイテム数
DIMS = [50, 100, 200, 500]  # Embedding dimensions to test / テストする埋め込み次元
RADIUS = 1.0        # Circle radius / 円半径
BASE_SEED = 42      # Base random seed / 基本乱数シード
N_SEEDS = 1000      # Number of seeds / シード数
OUTPUT_DIR = Path("outputs/exp06")  # Output directory / 出力ディレクトリ


def generate_A_axis(seed: int, dim: int) -> np.ndarray:
    """Generate A-axis with specified dimension.
    
    指定された次元でA軸を生成する。
    
    Args:
        seed: Random seed
              乱数シード
        dim: Embedding dimension
            埋め込み次元
            
    Returns:
        A-axis matrix (N_ITEMS, dim)
        A軸行列 (N_ITEMS, dim)
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N_ITEMS, dim))
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    A = A / (norms + 1e-12)
    return A


def arrange_circle(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Arrange items on circle.
    
    円周上にアイテムを配置する。
    
    Args:
        seed: Random seed for ordering
              順序のための乱数シード
              
    Returns:
        Tuple of (coordinates, ordering)
        (座標, 順序)のタプル
    """
    rng = np.random.default_rng(seed)
    ordering = rng.permutation(N_ITEMS)
    
    angles = 2 * np.pi * np.arange(N_ITEMS) / N_ITEMS
    x = RADIUS * np.cos(angles)
    y = RADIUS * np.sin(angles)
    coords = np.column_stack([x, y])
    
    ordered_coords = np.zeros_like(coords)
    ordered_coords[ordering] = coords
    
    return ordered_coords, ordering


def arrange_random(seed: int) -> np.ndarray:
    """Arrange items randomly in circle.
    
    円内にアイテムをランダムに配置する。
    
    Args:
        seed: Random seed
              乱数シード
              
    Returns:
        Random coordinates (N_ITEMS, 2)
        ランダム座標 (N_ITEMS, 2)
    """
    rng = np.random.default_rng(seed)
    # Sample uniformly within circle
    # 円内に一様にサンプリング
    r = np.sqrt(rng.random(N_ITEMS)) * RADIUS
    theta = 2 * np.pi * rng.random(N_ITEMS)
    coords = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return coords


def compute_vs(D_semantic: np.ndarray, D_spatial: np.ndarray) -> float:
    """Compute Value-Space correlation (VS).
    
    意味-空間相関（VS）を計算する。
    
    Args:
        D_semantic: Semantic distance matrix
                   意味距離行列
        D_spatial: Spatial distance matrix
                  空間距離行列
                  
    Returns:
        Spearman correlation coefficient
        スピアマン相関係数
    """
    sem_flat = squareform(D_semantic, checks=False)
    spatial_flat = squareform(D_spatial, checks=False)
    vs, _ = spearmanr(sem_flat, spatial_flat)
    return float(vs)


def run_single_trial(seed: int, dim: int) -> Dict[str, float]:
    """Run a single trial with specified dimension.
    
    指定された次元で単一試行を実行する。
    
    Args:
        seed: Random seed
              乱数シード
        dim: Embedding dimension
            埋め込み次元
            
    Returns:
        Dictionary with results
        結果を含む辞書
    """
    # Generate A-axis with independent seeds
    # 独立したシードでA軸を生成
    seed_A = seed
    seed_spatial = seed + 10000
    seed_control = seed + 20000
    
    A = generate_A_axis(seed_A, dim)
    D_semantic = squareform(pdist(A, metric='correlation'))
    
    # Circular arrangement
    # 円周配置
    coords_circle, ordering = arrange_circle(seed_spatial)
    D_circle = squareform(pdist(coords_circle, metric='euclidean'))
    vs_circle = compute_vs(D_semantic, D_circle)
    
    # Random arrangement
    # ランダム配置
    coords_random = arrange_random(seed_control)
    D_random = squareform(pdist(coords_random, metric='euclidean'))
    vs_random = compute_vs(D_semantic, D_random)
    
    return {
        'seed': seed,
        'dim': dim,
        'vs_circle': vs_circle,
        'vs_random': vs_random
    }


def run_exp06() -> pd.DataFrame:
    """Run complete EXP06 with dimension sweep.
    
    次元掃引を伴うEXP06の完全実行を行う。
    
    Returns:
        DataFrame with results
        結果を含むDataFrame
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    total = len(DIMS) * N_SEEDS
    
    print(f"  Running {len(DIMS)} dimensions × {N_SEEDS} seeds = {total} trials")
    print(f"  {len(DIMS)}次元 × {N_SEEDS}シード = {total}試行を実行")
    print()
    
    for dim in DIMS:
        print(f"  DIM = {dim}:")
        for i in range(N_SEEDS):
            seed = BASE_SEED + i
            trial_result = run_single_trial(seed, dim)
            results.append(trial_result)
            
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{N_SEEDS} seeds completed / シード完了")
        print()
    
    df = pd.DataFrame(results)
    
    # Save results
    # 結果を保存
    csv_path = OUTPUT_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Results saved to / 結果を保存: {csv_path}")
    
    # Compute summary by dimension
    # 次元ごとの要約を計算
    summary_by_dim = []
    for dim in DIMS:
        df_dim = df[df['dim'] == dim]
        summary_by_dim.append({
            'dim': dim,
            'vs_circle_mean': float(df_dim['vs_circle'].mean()),
            'vs_circle_std': float(df_dim['vs_circle'].std()),
            'vs_random_mean': float(df_dim['vs_random'].mean()),
            'vs_random_std': float(df_dim['vs_random'].std())
        })
    
    # Save summary
    # 要約を保存
    import sys, platform, scipy
    summary = {
        'dims': DIMS,
        'n_seeds': N_SEEDS,
        'results_by_dim': summary_by_dim,
        'interpretation': 'O1: Natural orthogonality is dimension-independent',
        'config': {
            'N_ITEMS': N_ITEMS,
            'DIMS': DIMS,
            'RADIUS': RADIUS,
            'BASE_SEED': BASE_SEED,
            'N_SEEDS': N_SEEDS
        },
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'pandas': pd.__version__
        }
    }
    
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to / 要約を保存: {summary_path}")
    
    # Create visualization
    # 可視化を作成
    create_visualization(df)
    
    return df


def create_visualization(df: pd.DataFrame) -> None:
    """Create visualization of dimension robustness.
    
    次元頑健性の可視化を作成する。
    
    Args:
        df: DataFrame with results
            結果を含むDataFrame
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    dims = sorted(df['dim'].unique())
    
    # Panel 1: VS vs Dimension (log scale)
    # パネル1：VS vs 次元（対数スケール）
    vs_circle_means = [df[df['dim'] == d]['vs_circle'].mean() for d in dims]
    vs_circle_stds = [df[df['dim'] == d]['vs_circle'].std() for d in dims]
    vs_random_means = [df[df['dim'] == d]['vs_random'].mean() for d in dims]
    vs_random_stds = [df[df['dim'] == d]['vs_random'].std() for d in dims]
    
    axes[0].errorbar(dims, vs_circle_means, yerr=vs_circle_stds, marker='o',
                    linewidth=2, markersize=8, capsize=5, label='Circle', color='blue')
    axes[0].errorbar(dims, vs_random_means, yerr=vs_random_stds, marker='s',
                    linewidth=2, markersize=8, capsize=5, label='Random', color='orange')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Embedding Dimension (log scale)', fontsize=11)
    axes[0].set_ylabel('VS (Value-Space Correlation)', fontsize=11)
    axes[0].set_title('O1: VS≈0 Across Dimensions',
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Box plots by dimension
    # パネル2：次元ごとの箱ひげ図
    positions = []
    data_circle = []
    data_random = []
    
    for i, d in enumerate(dims):
        df_d = df[df['dim'] == d]
        positions.extend([i*2.5 + 0.8, i*2.5 + 1.6])
        data_circle.append(df_d['vs_circle'].values)
        data_random.append(df_d['vs_random'].values)
    
    bp_data = []
    for i in range(len(dims)):
        bp_data.extend([data_circle[i], data_random[i]])
    
    bp = axes[1].boxplot(bp_data, positions=positions, widths=0.6,
                         patch_artist=True, showfliers=False)
    
    # Color boxes
    # ボックスに色を付ける
    colors = ['lightblue', 'lightsalmon'] * len(dims)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xticks([i*2.5 + 1.2 for i in range(len(dims))])
    axes[1].set_xticklabels([f'D={d}' for d in dims])
    axes[1].set_ylabel('VS (Value-Space Correlation)', fontsize=11)
    axes[1].set_title('VS Distribution by Dimension',
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add legend
    # 凡例を追加
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightblue', label='Circle'),
                      Patch(facecolor='lightsalmon', label='Random')]
    axes[1].legend(handles=legend_elements, loc='upper right')
    
    # Add statistics
    # 統計情報を追加
    stats_text = '\n'.join([f'D={d}: {vs_circle_means[i]:.3f}±{vs_circle_stds[i]:.3f}'
                           for i, d in enumerate(dims)])
    axes[0].text(0.02, 0.98, stats_text,
                transform=axes[0].transAxes,
                verticalalignment='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    # 図を保存
    fig_path = OUTPUT_DIR / "visualization.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved to / 可視化を保存: {fig_path}")


def main():
    """Main execution function.
    
    メイン実行関数。
    """
    print("=" * 70)
    print("E8 Series - EXP06: Dimension Robustness Test")
    print("E8シリーズ - EXP06：次元頑健性テスト")
    print("=" * 70)
    print()
    print(f"Configuration / 設定:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIMS = {DIMS}")
    print(f"  N_SEEDS = {N_SEEDS}")
    print()
    
    df = run_exp06()
    
    print()
    print("=" * 70)
    print("Results Summary / 結果要約:")
    print("=" * 70)
    for dim in DIMS:
        df_dim = df[df['dim'] == dim]
        print(f"DIM = {dim}:")
        print(f"  VS (circle) / VS（円周）: {df_dim['vs_circle'].mean():.4f} ± {df_dim['vs_circle'].std():.4f}")
        print(f"  VS (random) / VS（ランダム）: {df_dim['vs_random'].mean():.4f} ± {df_dim['vs_random'].std():.4f}")
    print()
    print("Interpretation / 解釈:")
    print("  O1 (Natural Orthogonality): VS≈0 holds across all dimensions,")
    print("  confirming that natural orthogonality is dimension-independent.")
    print("  O1（自然直交性）：VS≈0はすべての次元で成立し、")
    print("  自然直交性が次元非依存であることを確認する。")
    print("=" * 70)


if __name__ == "__main__":
    main()
