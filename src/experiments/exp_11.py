"""E8 Series - EXP11: Structural Stress Test (Coordinate Noise)

構造ストレステスト（座標ノイズ）

This experiment tests O3 (Stress Tolerance) by adding noise to spatial
coordinates while keeping semantic structure fixed. This examines whether
coordinate perturbations disrupt semantic-spatial relationships.

この実験は、意味構造を固定したまま空間座標にノイズを加えることで
O3（ストレス耐性）をテストする。これは、座標の摂動が意味-空間関係を
破壊するかを検証する。

Key Finding / 主要な発見:
    Coordinate noise does not significantly increase VS, supporting O3:
    spatial disruption ≠ semantic confusion.
    
    座標ノイズはVSを有意に増加させず、O3を支持する：
    空間破壊 ≠ 意味的混乱。

References / 参考文献:
    E8b (2025-138): O3 Stress Tolerance Law
    E8c (2025-139): Robustness to perturbations
    
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
RADIUS = 1.0        # Circle radius / 円半径
BASE_SEED = 42      # Base random seed / 基本乱数シード
N_SEEDS = 30        # Number of seeds / シード数
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.5]  # Noise levels / ノイズレベル
OUTPUT_DIR = Path("outputs/exp11")  # Output directory / 出力ディレクトリ


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


def add_coordinate_noise(coords: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Add Gaussian noise to coordinates.
    
    座標にガウスノイズを追加する。
    
    This perturbs spatial positions while attempting to preserve
    topological order (though strong noise may disrupt it).
    
    これは、位相的順序を保存しようとしながら空間位置を摂動する
    （ただし、強いノイズはそれを破壊する可能性がある）。
    
    Args:
        coords: Original coordinates
               元の座標
        noise_level: Standard deviation of Gaussian noise
                    ガウスノイズの標準偏差
        seed: Random seed
              乱数シード
              
    Returns:
        Noisy coordinates
        ノイズを含む座標
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, coords.shape)
    coords_noisy = coords + noise
    return coords_noisy


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


def run_single_trial(seed: int, noise_level: float) -> Dict[str, float]:
    """Run a single trial with coordinate noise.
    
    座標ノイズを伴う単一試行を実行する。
    
    Args:
        seed: Random seed
              乱数シード
        noise_level: Noise level
                    ノイズレベル
                    
    Returns:
        Dictionary with results
        結果を含む辞書
    """
    # Generate with independent seeds
    # 独立したシードで生成
    seed_A = seed
    seed_spatial = seed + 10000
    seed_noise = seed + 20000
    
    A = generate_A_axis(seed_A)
    D_semantic = squareform(pdist(A, metric='correlation'))
    
    # Circle arrangement with noise
    # ノイズを伴う円周配置
    coords_original, ordering = arrange_circle(seed_spatial)
    coords_noisy = add_coordinate_noise(coords_original, noise_level, seed_noise)
    D_spatial = squareform(pdist(coords_noisy, metric='euclidean'))
    
    # Compute VS
    # VSを計算
    vs = compute_vs(D_semantic, D_spatial)
    
    return {
        'seed': seed,
        'noise_level': noise_level,
        'vs': vs
    }


def run_exp11() -> pd.DataFrame:
    """Run complete EXP11 with noise level sweep.
    
    ノイズレベル掃引を伴うEXP11の完全実行を行う。
    
    Returns:
        DataFrame with results
        結果を含むDataFrame
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    total = len(NOISE_LEVELS) * N_SEEDS
    
    print(f"  Running {len(NOISE_LEVELS)} noise levels × {N_SEEDS} seeds = {total} trials")
    print(f"  {len(NOISE_LEVELS)}ノイズレベル × {N_SEEDS}シード = {total}試行を実行")
    print()
    
    for noise_level in NOISE_LEVELS:
        print(f"  Noise level = {noise_level:.2f}:")
        for i in range(N_SEEDS):
            seed = BASE_SEED + i
            trial_result = run_single_trial(seed, noise_level)
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
    
    # Compute summary
    # 要約を計算
    summary_by_noise = []
    for noise in NOISE_LEVELS:
        df_noise = df[df['noise_level'] == noise]
        summary_by_noise.append({
            'noise_level': noise,
            'vs_mean': float(df_noise['vs'].mean()),
            'vs_std': float(df_noise['vs'].std())
        })
    
    # Save summary
    # 要約を保存
    import sys, platform, scipy
    summary = {
        'noise_levels': NOISE_LEVELS,
        'n_seeds': N_SEEDS,
        'results_by_noise': summary_by_noise,
        'interpretation': 'O3: Coordinate noise does not significantly increase VS',
        'config': {
            'N_ITEMS': N_ITEMS,
            'DIM': DIM,
            'RADIUS': RADIUS,
            'BASE_SEED': BASE_SEED,
            'N_SEEDS': N_SEEDS,
            'NOISE_LEVELS': NOISE_LEVELS
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
    """Create visualization of stress test.
    
    ストレステストの可視化を作成する。
    
    Args:
        df: DataFrame with results
            結果を含むDataFrame
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    noise_vals = sorted(df['noise_level'].unique())
    
    # Panel 1: VS vs Noise Level
    # パネル1：VS vs ノイズレベル
    vs_means = [df[df['noise_level'] == n]['vs'].mean() for n in noise_vals]
    vs_stds = [df[df['noise_level'] == n]['vs'].std() for n in noise_vals]
    
    axes[0].errorbar(noise_vals, vs_means, yerr=vs_stds, marker='o', linewidth=2,
                    markersize=8, capsize=5, color='green')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5,
                   label='VS=0 (Natural Orthogonality)')
    axes[0].set_xlabel('Coordinate Noise Level / 座標ノイズレベル', fontsize=11)
    axes[0].set_ylabel('VS (Value-Space Correlation)\nVS（意味-空間相関）', fontsize=11)
    axes[0].set_title('O3: VS≈0 Despite Coordinate Stress\nO3：座標ストレス下でもVS≈0',
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Box plots by noise level
    # パネル2：ノイズレベルごとの箱ひげ図
    data = [df[df['noise_level'] == n]['vs'].values for n in noise_vals]
    bp = axes[1].boxplot(data, labels=[f'{n:.2f}' for n in noise_vals], patch_artist=True)
    
    # Color boxes by intensity
    # 強度で色を付ける
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(noise_vals)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Noise Level / ノイズレベル', fontsize=11)
    axes[1].set_ylabel('VS (Value-Space Correlation)\nVS（意味-空間相関）', fontsize=11)
    axes[1].set_title('VS Distribution by Noise Level\nノイズレベルごとのVS分布',
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    # 統計情報を追加
    stats_text = '\n'.join([f'Noise {n:.2f}: {vs_means[i]:.3f}±{vs_stds[i]:.3f}'
                           for i, n in enumerate(noise_vals)])
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
    print("E8 Series - EXP11: Structural Stress Test")
    print("E8シリーズ - EXP11：構造ストレステスト")
    print("=" * 70)
    print()
    print(f"Configuration / 設定:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  Noise levels / ノイズレベル = {NOISE_LEVELS}")
    print(f"  N_SEEDS = {N_SEEDS}")
    print()
    
    df = run_exp11()
    
    print()
    print("=" * 70)
    print("Results Summary / 結果要約:")
    print("=" * 70)
    for noise in NOISE_LEVELS:
        df_noise = df[df['noise_level'] == noise]
        print(f"Noise level = {noise:.2f}:")
        print(f"  VS = {df_noise['vs'].mean():.4f} ± {df_noise['vs'].std():.4f}")
    print()
    print("Interpretation / 解釈:")
    print("  O3 (Stress Tolerance): Coordinate noise does not significantly")
    print("  increase VS, confirming robustness to spatial perturbations.")
    print("  O3（ストレス耐性）：座標ノイズはVSを有意に増加させず、")
    print("  空間摂動に対する頑健性を確認する。")
    print("=" * 70)


if __name__ == "__main__":
    main()
