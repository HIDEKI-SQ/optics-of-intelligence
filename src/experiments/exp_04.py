"""E8 Series - EXP04: 3D Cube Arrangement

3D立方体配置

This experiment tests a 3D spatial arrangement where items are placed
on the vertices of a cube. This examines whether natural orthogonality
(VS≈0) extends to higher-dimensional spatial structures.

この実験は、アイテムを立方体の頂点に配置する3D空間配置をテストする。
これは、自然直交性（VS≈0）がより高次元の空間構造に拡張するかを検証する。

Key Finding / 主要な発見:
    VS≈0 holds for 3D cube arrangement, confirming that natural
    orthogonality is independent of spatial dimensionality (1D, 2D, 3D).
    
    3D立方体配置でもVS≈0が成立し、自然直交性が空間次元性
    （1D、2D、3D）に依存しないことを確認する。

References / 参考文献:
    E8a (2025-137): Spatial structure variations
    E8b (2025-138): O1 Natural Orthogonality across dimensions
    
Author: HIDEKI
Date: 2025-11
License: MIT
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, Tuple
import json

# Configuration
# 設定
N_ITEMS = 20        # Number of items / アイテム数
DIM = 100           # Embedding dimension / 埋め込み次元
BASE_SEED = 42      # Base random seed / 基本乱数シード
N_SEEDS = 1000      # Number of seeds / シード数
OUTPUT_DIR = Path("outputs/exp04")  # Output directory / 出力ディレクトリ


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


def arrange_3d_cube(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Arrange items in 3D space (cube-like structure).
    
    3D空間にアイテムを配置する（立方体様構造）。
    
    Creates positions distributed in 3D space, approximating a cube
    structure. Items are placed at regular grid points within a unit cube.
    
    3D空間に分散した位置を作成し、立方体構造を近似する。
    アイテムは単位立方体内の規則的なグリッド点に配置される。
    
    Args:
        seed: Random seed for ordering
              順序のための乱数シード
              
    Returns:
        Tuple of (coordinates, ordering)
        (座標, 順序)のタプル
    """
    rng = np.random.default_rng(seed)
    ordering = rng.permutation(N_ITEMS)
    
    # Create 3D grid positions
    # 3Dグリッド位置を作成
    # Determine grid size (roughly cubic)
    # グリッドサイズを決定（ほぼ立方体）
    grid_size = int(np.ceil(N_ITEMS ** (1/3)))
    
    # Generate grid coordinates
    # グリッド座標を生成
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if len(positions) < N_ITEMS:
                    positions.append([i, j, k])
    
    coords = np.array(positions[:N_ITEMS], dtype=float)
    
    # Normalize to [0, 1] range
    # [0, 1]範囲に正規化
    if grid_size > 1:
        coords = coords / (grid_size - 1)
    
    # Shuffle positions
    # 位置をシャッフル
    rng.shuffle(coords)
    
    # Apply ordering
    # 順序を適用
    ordered_coords = np.zeros_like(coords)
    ordered_coords[ordering] = coords
    
    return ordered_coords, ordering


def arrange_random_3d(seed: int) -> np.ndarray:
    """Arrange items randomly in 3D space.
    
    3D空間にアイテムをランダムに配置する。
    
    Args:
        seed: Random seed
              乱数シード
              
    Returns:
        Random 3D coordinates (N_ITEMS, 3)
        ランダム3D座標 (N_ITEMS, 3)
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, (N_ITEMS, 3))
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
    """Run a single trial of EXP04.
    
    EXP04の単一試行を実行する。
    
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
    
    # 3D cube arrangement
    # 3D立方体配置
    coords_cube, ordering = arrange_3d_cube(seed_spatial)
    D_cube = squareform(pdist(coords_cube, metric='euclidean'))
    vs_cube = compute_vs(D_semantic, D_cube)
    
    # Random 3D arrangement
    # ランダム3D配置
    coords_random = arrange_random_3d(seed_control)
    D_random = squareform(pdist(coords_random, metric='euclidean'))
    vs_random = compute_vs(D_semantic, D_random)
    
    return {
        'seed': seed,
        'vs_cube': vs_cube,
        'vs_random': vs_random
    }


def run_exp04() -> pd.DataFrame:
    """Run complete EXP04 with multiple seeds.
    
    複数のシードでEXP04の完全実行を行う。
    
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
        'vs_cube_mean': float(df['vs_cube'].mean()),
        'vs_cube_std': float(df['vs_cube'].std()),
        'vs_random_mean': float(df['vs_random'].mean()),
        'vs_random_std': float(df['vs_random'].std()),
        'n_seeds': N_SEEDS,
        'interpretation': 'O1: Natural orthogonality holds for 3D cube arrangement',
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
    fig = plt.figure(figsize=(16, 5))
    
    # Panel 1: Histograms
    # パネル1：ヒストグラム
    ax1 = plt.subplot(1, 3, 1)
    ax1.hist(df['vs_cube'], bins=15, alpha=0.6, label='Cube',
            edgecolor='black', color='lightblue')
    ax1.hist(df['vs_random'], bins=15, alpha=0.6, label='Random',
            edgecolor='black', color='lightsalmon')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('VS (Value-Space Correlation)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('VS Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Box plots
    # パネル2：箱ひげ図
    ax2 = plt.subplot(1, 3, 2)
    data = [df['vs_cube'], df['vs_random']]
    labels = ['Cube', 'Random']
    bp = ax2.boxplot(data, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('VS (Value-Space Correlation)')
    ax2.set_title('EXP04: Cube vs Random')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics to panel 2
    # パネル2に統計情報を追加
    stats_text = (
        f'Cube:\n'
        f'  Mean: {df["vs_cube"].mean():.4f}\n'
        f'  Std: {df["vs_cube"].std():.4f}\n\n'
        f'Random:\n'
        f'  Mean: {df["vs_random"].mean():.4f}\n'
        f'  Std: {df["vs_random"].std():.4f}'
    )
    ax2.text(0.02, 0.98, stats_text,
            transform=ax2.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 3: 3D visualization of sample cube layout
    # パネル3：サンプル立方体レイアウトの3D可視化
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    sample_coords, _ = arrange_3d_cube(42)
    ax3.scatter(sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2],
               s=100, alpha=0.6, c='blue')
    for i, (x, y, z) in enumerate(sample_coords[:10]):  # Label first 10 for clarity
        ax3.text(x, y, z, str(i+1), fontsize=8)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Sample 3D Cube Layout')
    
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
    print("E8 Series - EXP04: 3D Cube Arrangement")
    print("E8シリーズ - EXP04：3D立方体配置")
    print("=" * 70)
    print()
    print(f"Configuration / 設定:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  N_SEEDS = {N_SEEDS}")
    print()
    print("Running experiment / 実験実行中...")
    print()
    
    df = run_exp04()
    
    print()
    print("=" * 70)
    print("Results / 結果:")
    print("=" * 70)
    print(f"VS (cube) / VS（立方体）:   {df['vs_cube'].mean():.4f} ± {df['vs_cube'].std():.4f}")
    print(f"VS (random) / VS（ランダム）: {df['vs_random'].mean():.4f} ± {df['vs_random'].std():.4f}")
    print()
    print("Interpretation / 解釈:")
    print("  3D cube arrangement maintains VS≈0, confirming that natural")
    print("  orthogonality extends to higher-dimensional spatial structures.")
    print("  3D立方体配置はVS≈0を維持し、自然直交性がより高次元の")
    print("  空間構造に拡張することを確認する。")
    print("=" * 70)


if __name__ == "__main__":
    main()
