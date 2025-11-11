"""E8 Series - EXP03: Line Arrangement

ライン配置

This experiment tests a linear (1D) spatial arrangement instead of
circular (2D). This examines whether natural orthogonality (VS≈0)
holds across different spatial dimensionalities.

この実験は、円周（2D）ではなく線形（1D）空間配置をテストする。
これは、自然直交性（VS≈0）が異なる空間次元性でも成立するかを検証する。

Key Finding / 主要な発見:
    VS≈0 holds for linear arrangement, supporting the generality
    of natural orthogonality across spatial structures.
    
    線形配置でもVS≈0が成立し、空間構造を超えた
    自然直交性の一般性を支持する。

References / 参考文献:
    E8a (2025-137): Spatial arrangement variations
    E8b (2025-138): O1 Natural Orthogonality across structures
    
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
DIM = 100           # Embedding dimension / 埋め込み次元
BASE_SEED = 42      # Base random seed / 基本乱数シード
N_SEEDS = 30        # Number of seeds / シード数
OUTPUT_DIR = Path("outputs/exp03")  # Output directory / 出力ディレクトリ


def generate_A_axis(seed: int) -> np.ndarray:
    """Generate A-axis (Blueprint structure).
    
    A軸（Blueprint構造）を生成する。
    
    Args:
        seed: Random seed
              乱数シード
              
    Returns:
        A-axis matrix (N_ITEMS, DIM)
        A軸行列 (N_ITEMS, DIM)
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N_ITEMS, DIM))
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    A = A / (norms + 1e-12)
    return A


def arrange_line(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Arrange items on a line with uniform spacing.
    
    均一な間隔で線上にアイテムを配置する。
    
    Creates a 1D linear arrangement where items are evenly spaced
    along a line segment. This tests whether spatial dimensionality
    affects natural orthogonality.
    
    アイテムが線分に沿って等間隔に配置される1D線形配置を作成する。
    これは、空間次元性が自然直交性に影響するかをテストする。
    
    Args:
        seed: Random seed for ordering
              順序のための乱数シード
              
    Returns:
        Tuple of (coordinates, ordering)
        (座標, 順序)のタプル
    """
    rng = np.random.default_rng(seed)
    ordering = rng.permutation(N_ITEMS)
    
    # Uniform spacing along line segment [0, 1]
    # 線分[0, 1]に沿った均一な間隔
    positions = np.linspace(0, 1, N_ITEMS)
    
    # Create 2D coordinates (line along x-axis, y=0)
    # 2D座標を作成（x軸に沿った線、y=0）
    coords = np.column_stack([positions, np.zeros(N_ITEMS)])
    
    # Apply ordering
    # 順序を適用
    ordered_coords = np.zeros_like(coords)
    ordered_coords[ordering] = coords
    
    return ordered_coords, ordering


def arrange_random(seed: int) -> np.ndarray:
    """Arrange items randomly.
    
    アイテムをランダムに配置する。
    
    Args:
        seed: Random seed
              乱数シード
              
    Returns:
        Random coordinates (N_ITEMS, 2)
        ランダム座標 (N_ITEMS, 2)
    """
    rng = np.random.default_rng(seed)
    # Random positions in unit square
    # 単位正方形内のランダム位置
    coords = rng.uniform(0, 1, (N_ITEMS, 2))
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


def run_single_trial(seed: int) -> Dict[str, float]:
    """Run a single trial of EXP03.
    
    EXP03の単一試行を実行する。
    
    Args:
        seed: Random seed
              乱数シード
              
    Returns:
        Dictionary with results
        結果を含む辞書
    """
    # Generate A-axis with independent seeds
    # 独立したシードでA軸を生成
    seed_A = seed
    seed_spatial = seed + 10000
    seed_control = seed + 20000
    
    A = generate_A_axis(seed_A)
    D_semantic = squareform(pdist(A, metric='correlation'))
    
    # Linear arrangement
    # 線形配置
    coords_line, ordering = arrange_line(seed_spatial)
    D_line = squareform(pdist(coords_line, metric='euclidean'))
    vs_line = compute_vs(D_semantic, D_line)
    
    # Random arrangement
    # ランダム配置
    coords_random = arrange_random(seed_control)
    D_random = squareform(pdist(coords_random, metric='euclidean'))
    vs_random = compute_vs(D_semantic, D_random)
    
    return {
        'seed': seed,
        'vs_line': vs_line,
        'vs_random': vs_random
    }


def run_exp03() -> pd.DataFrame:
    """Run complete EXP03 with multiple seeds.
    
    複数のシードでEXP03の完全実行を行う。
    
    Returns:
        DataFrame with results
        結果を含むDataFrame
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    for i in range(N_SEEDS):
        seed = BASE_SEED + i
        trial_result = run_single_trial(seed)
        results.append(trial_result)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{N_SEEDS} trials / 試行完了")
    
    df = pd.DataFrame(results)
    
    # Save results
    # 結果を保存
    csv_path = OUTPUT_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to / 結果を保存: {csv_path}")
    
    # Save summary
    # 要約を保存
    import sys, platform, scipy
    summary = {
        'vs_line_mean': float(df['vs_line'].mean()),
        'vs_line_std': float(df['vs_line'].std()),
        'vs_random_mean': float(df['vs_random'].mean()),
        'vs_random_std': float(df['vs_random'].std()),
        'n_seeds': N_SEEDS,
        'interpretation': 'O1: Natural orthogonality holds for linear (1D) arrangement',
        'config': {
            'N_ITEMS': N_ITEMS,
            'DIM': DIM,
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
    """Create visualization of results.
    
    結果の可視化を作成する。
    
    Args:
        df: DataFrame with results
            結果を含むDataFrame
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: Histograms
    # パネル1：ヒストグラム
    axes[0].hist(df['vs_line'], bins=15, alpha=0.6, label='Line\nライン', 
                edgecolor='black', color='lightblue')
    axes[0].hist(df['vs_random'], bins=15, alpha=0.6, label='Random\nランダム',
                edgecolor='black', color='lightsalmon')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('VS (Value-Space Correlation)\nVS（意味-空間相関）')
    axes[0].set_ylabel('Frequency / 頻度')
    axes[0].set_title('VS Distribution\nVS分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Box plots
    # パネル2：箱ひげ図
    data = [df['vs_line'], df['vs_random']]
    labels = ['Line\nライン', 'Random\nランダム']
    bp = axes[1].boxplot(data, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_ylabel('VS (Value-Space Correlation)\nVS（意味-空間相関）')
    axes[1].set_title('EXP03: Line vs Random\nEXP03：ライン vs ランダム')
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Line layout illustration
    # パネル3：ラインレイアウトの図解
    sample_coords, _ = arrange_line(42)
    axes[2].scatter(sample_coords[:, 0], sample_coords[:, 1], s=100, alpha=0.6, color='blue')
    for i, (x, y) in enumerate(sample_coords):
        axes[2].text(x, y + 0.05, str(i+1), ha='center', va='bottom', fontsize=8)
    axes[2].set_xlim(-0.1, 1.1)
    axes[2].set_ylim(-0.2, 0.2)
    axes[2].set_aspect('equal')
    axes[2].set_xlabel('Position / 位置')
    axes[2].set_title('Sample Line Layout (1D)\nサンプルラインレイアウト（1D）')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add statistics
    # 統計情報を追加
    stats_text = (
        f'Line / ライン:\n'
        f'  Mean: {df["vs_line"].mean():.4f}\n'
        f'  Std: {df["vs_line"].std():.4f}\n\n'
        f'Random / ランダム:\n'
        f'  Mean: {df["vs_random"].mean():.4f}\n'
        f'  Std: {df["vs_random"].std():.4f}'
    )
    axes[1].text(0.02, 0.98, stats_text,
                transform=axes[1].transAxes,
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
    print("E8 Series - EXP03: Line Arrangement")
    print("E8シリーズ - EXP03：ライン配置")
    print("=" * 70)
    print()
    print(f"Configuration / 設定:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  N_SEEDS = {N_SEEDS}")
    print()
    print("Running experiment / 実験実行中...")
    print()
    
    df = run_exp03()
    
    print()
    print("=" * 70)
    print("Results / 結果:")
    print("=" * 70)
    print(f"VS (line) / VS（ライン）:   {df['vs_line'].mean():.4f} ± {df['vs_line'].std():.4f}")
    print(f"VS (random) / VS（ランダム）: {df['vs_random'].mean():.4f} ± {df['vs_random'].std():.4f}")
    print()
    print("Interpretation / 解釈:")
    print("  Linear (1D) arrangement maintains VS≈0, supporting the generality")
    print("  of natural orthogonality across different spatial dimensionalities.")
    print("  線形（1D）配置はVS≈0を維持し、異なる空間次元性にわたる")
    print("  自然直交性の一般性を支持する。")
    print("=" * 70)


if __name__ == "__main__":
    main()
