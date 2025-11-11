"""E8 Series - EXP02: Grid Arrangement

グリッド配置

This experiment tests a structured grid arrangement as an alternative
to circular spatial layout. This examines whether different spatial
structures maintain natural orthogonality (VS≈0).

この実験は、円周空間配置の代替として構造化されたグリッド配置をテストする。
これは、異なる空間構造が自然直交性（VS≈0）を維持するかを検証する。

Key Finding / 主要な発見:
    Grid arrangement also exhibits VS≈0, supporting the generality
    of natural orthogonality across different spatial structures.
    
    グリッド配置もVS≈0を示し、異なる空間構造にわたる
    自然直交性の一般性を支持する。

References / 参考文献:
    E8a (2025-137): Initial spatial arrangements
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
from typing import Dict
import json

# Configuration
# 設定
N_ITEMS = 20        # Number of items / アイテム数
DIM = 100           # Embedding dimension / 埋め込み次元
BASE_SEED = 42      # Base random seed / 基本乱数シード
N_SEEDS = 30        # Number of seeds / シード数
OUTPUT_DIR = Path("outputs/exp02")  # Output directory / 出力ディレクトリ


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


def arrange_grid(seed: int) -> np.ndarray:
    """Arrange items on a regular grid.
    
    アイテムを規則的なグリッド上に配置する。
    
    Creates a roughly square grid and randomly assigns items to positions.
    This preserves spatial structure (regularity) while avoiding semantic bias.
    
    ほぼ正方形のグリッドを作成し、アイテムをランダムに位置に割り当てる。
    これは意味的バイアスを避けながら空間構造（規則性）を保存する。
    
    Args:
        seed: Random seed
              乱数シード
              
    Returns:
        Grid coordinates (N_ITEMS, 2)
        グリッド座標 (N_ITEMS, 2)
    """
    rng = np.random.default_rng(seed)
    
    # Determine grid dimensions (roughly square)
    # グリッド次元を決定（ほぼ正方形）
    grid_size = int(np.ceil(np.sqrt(N_ITEMS)))
    
    # Create grid positions
    # グリッド位置を作成
    x_positions = np.arange(grid_size)
    y_positions = np.arange(grid_size)
    xx, yy = np.meshgrid(x_positions, y_positions)
    grid_positions = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Take first N_ITEMS positions and shuffle
    # 最初のN_ITEMS位置を取得してシャッフル
    grid_positions = grid_positions[:N_ITEMS]
    rng.shuffle(grid_positions)
    
    # Normalize to [-1, 1] range
    # [-1, 1]範囲に正規化
    if grid_size > 1:
        grid_positions = 2 * grid_positions / (grid_size - 1) - 1
    
    return grid_positions.astype(np.float64)


def arrange_random(seed: int) -> np.ndarray:
    """Arrange items randomly (control condition).
    
    アイテムをランダムに配置する（統制条件）。
    
    Args:
        seed: Random seed
              乱数シード
              
    Returns:
        Random coordinates (N_ITEMS, 2)
        ランダム座標 (N_ITEMS, 2)
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-1, 1, (N_ITEMS, 2))
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
    """Run a single trial of EXP02.
    
    EXP02の単一試行を実行する。
    
    Args:
        seed: Random seed
              乱数シード
              
    Returns:
        Dictionary with VS values for both conditions
        両条件のVS値を含む辞書
    """
    # Generate A-axis
    # A軸を生成
    seed_A = seed
    seed_spatial = seed + 10000
    A = generate_A_axis(seed_A)
    
    # Compute semantic distances
    # 意味距離を計算
    D_semantic = squareform(pdist(A, metric='correlation'))
    
    # Grid arrangement
    # グリッド配置
    coords_grid = arrange_grid(seed_spatial)
    D_grid = squareform(pdist(coords_grid, metric='euclidean'))
    vs_grid = compute_vs(D_semantic, D_grid)
    
    # Random arrangement (control)
    # ランダム配置（統制）
    coords_random = arrange_random(seed_spatial)
    D_random = squareform(pdist(coords_random, metric='euclidean'))
    vs_random = compute_vs(D_semantic, D_random)
    
    return {
        'seed': seed,
        'vs_grid': vs_grid,
        'vs_random': vs_random
    }


def run_exp02() -> pd.DataFrame:
    """Run complete EXP02 with multiple seeds.
    
    複数のシードでEXP02の完全実行を行う。
    
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
        'vs_grid_mean': float(df['vs_grid'].mean()),
        'vs_grid_std': float(df['vs_grid'].std()),
        'vs_random_mean': float(df['vs_random'].mean()),
        'vs_random_std': float(df['vs_random'].std()),
        'n_seeds': N_SEEDS,
        'interpretation': 'Grid arrangement maintains VS≈0, supporting generality of O1',
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Histograms
    # パネル1：ヒストグラム
    axes[0].hist(df['vs_grid'], bins=15, alpha=0.6, label='Grid\nグリッド', edgecolor='black')
    axes[0].hist(df['vs_random'], bins=15, alpha=0.6, label='Random\nランダム', edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('VS (Value-Space Correlation)\nVS（意味-空間相関）')
    axes[0].set_ylabel('Frequency / 頻度')
    axes[0].set_title('VS Distribution\nVS分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Box plots
    # パネル2：箱ひげ図
    data = [df['vs_grid'], df['vs_random']]
    labels = ['Grid\nグリッド', 'Random\nランダム']
    axes[1].boxplot(data, labels=labels)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_ylabel('VS (Value-Space Correlation)\nVS（意味-空間相関）')
    axes[1].set_title('EXP02: Grid vs Random\nEXP02：グリッド vs ランダム')
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Grid layout illustration
    # パネル3：グリッドレイアウトの図解
    sample_grid = arrange_grid(42)
    axes[2].scatter(sample_grid[:, 0], sample_grid[:, 1], s=100, alpha=0.6)
    for i, (x, y) in enumerate(sample_grid):
        axes[2].text(x, y, str(i+1), ha='center', va='center', fontsize=8)
    axes[2].set_xlim(-1.2, 1.2)
    axes[2].set_ylim(-1.2, 1.2)
    axes[2].set_aspect('equal')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_title('Sample Grid Layout\nサンプルグリッドレイアウト')
    axes[2].grid(True, alpha=0.3)
    
    # Add statistics
    # 統計情報を追加
    stats_text = (
        f'Grid / グリッド:\n'
        f'  Mean: {df["vs_grid"].mean():.4f}\n'
        f'  Std: {df["vs_grid"].std():.4f}\n\n'
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
    print("E8 Series - EXP02: Grid Arrangement")
    print("E8シリーズ - EXP02：グリッド配置")
    print("=" * 70)
    print()
    print(f"Configuration / 設定:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  N_SEEDS = {N_SEEDS}")
    print()
    print("Running experiment / 実験実行中...")
    print()
    
    df = run_exp02()
    
    print()
    print("=" * 70)
    print("Results / 結果:")
    print("=" * 70)
    print(f"VS (grid) / VS（グリッド）:   {df['vs_grid'].mean():.4f} ± {df['vs_grid'].std():.4f}")
    print(f"VS (random) / VS（ランダム）: {df['vs_random'].mean():.4f} ± {df['vs_random'].std():.4f}")
    print()
    print("Interpretation / 解釈:")
    print("  Both grid and random arrangements show VS≈0.")
    print("  グリッドとランダム配置の両方がVS≈0を示す。")
    print()
    print("  This supports the generality of natural orthogonality (O1)")
    print("  across different spatial structures.")
    print("  これは、異なる空間構造にわたる自然直交性（O1）の一般性を支持する。")
    print("=" * 70)


if __name__ == "__main__":
    main()
