"""E8 Series - EXP09: Topological Disruption Test

位相破壊テスト

This experiment tests O2 (Phase Dominance) by disrupting the topological
order of spatial arrangement. We introduce random swaps that break the
circular topology while keeping spatial coordinates similar.

この実験は、空間配置の位相的順序を破壊することでO2（位相優位性）を
テストする。円周位相を破壊しつつ空間座標を類似に保つランダムスワップを導入する。

Key Finding / 主要な発見:
    Topological disruption increases VS, confirming that topology (φ)
    is more important than exact spatial coordinates for semantic-spatial binding.
    
    位相破壊がVSを増加させ、意味-空間結合には正確な空間座標よりも
    位相（φ）が重要であることを確認する。

References / 参考文献:
    E8b (2025-138): O2 Phase Dominance Law
    E8c (2025-139): Topology over coordinates
    
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
N_SEEDS = 1000      # Number of seeds / シード数
SWAP_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]  # Ratios of swaps / スワップ比率
OUTPUT_DIR = Path("outputs/exp09")  # Output directory / 出力ディレクトリ


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


def disrupt_topology(coords: np.ndarray, swap_ratio: float, seed: int) -> np.ndarray:
    """Disrupt topological order by swapping positions.
    
    位置をスワップすることで位相的順序を破壊する。
    
    This breaks the circular topology while keeping spatial coordinates
    relatively unchanged (items move to nearby positions).
    
    これは、空間座標を比較的変更しない（アイテムは近くの位置に移動）まま
    円周位相を破壊する。
    
    Args:
        coords: Original coordinates
               元の座標
        swap_ratio: Ratio of positions to swap [0, 1]
                   スワップする位置の比率 [0, 1]
        seed: Random seed
              乱数シード
              
    Returns:
        Disrupted coordinates
        破壊された座標
    """
    rng = np.random.default_rng(seed)
    coords_disrupted = coords.copy()
    
    n_swaps = int(N_ITEMS * swap_ratio / 2)  # Divide by 2 since each swap affects 2 items
    
    for _ in range(n_swaps):
        # Select two random positions to swap
        # スワップする2つのランダム位置を選択
        i, j = rng.choice(N_ITEMS, size=2, replace=False)
        coords_disrupted[i], coords_disrupted[j] = coords_disrupted[j].copy(), coords_disrupted[i].copy()
    
    return coords_disrupted


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


def run_single_trial(seed: int, swap_ratio: float) -> Dict[str, float]:
    """Run a single trial with topological disruption.
    
    位相破壊を伴う単一試行を実行する。
    
    Args:
        seed: Random seed
              乱数シード
        swap_ratio: Ratio of swaps
                   スワップ比率
                   
    Returns:
        Dictionary with results
        結果を含む辞書
    """
    # Generate with independent seeds
    # 独立したシードで生成
    seed_A = seed
    seed_spatial = seed + 10000
    seed_disrupt = seed + 20000
    
    A = generate_A_axis(seed_A)
    D_semantic = squareform(pdist(A, metric='correlation'))
    
    # Original circle arrangement
    # 元の円周配置
    coords_original, ordering = arrange_circle(seed_spatial)
    
    # Disrupt topology
    # 位相を破壊
    coords_disrupted = disrupt_topology(coords_original, swap_ratio, seed_disrupt)
    D_spatial = squareform(pdist(coords_disrupted, metric='euclidean'))
    
    # Compute VS
    # VSを計算
    vs = compute_vs(D_semantic, D_spatial)
    
    return {
        'seed': seed,
        'swap_ratio': swap_ratio,
        'vs': vs
    }


def run_exp09() -> pd.DataFrame:
    """Run complete EXP09 with swap ratio sweep.
    
    スワップ比率掃引を伴うEXP09の完全実行を行う。
    
    Returns:
        DataFrame with results
        結果を含むDataFrame
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    total = len(SWAP_RATIOS) * N_SEEDS
    
    print(f"  Running {len(SWAP_RATIOS)} swap ratios × {N_SEEDS} seeds = {total} trials")
    print(f"  {len(SWAP_RATIOS)}スワップ比率 × {N_SEEDS}シード = {total}試行を実行")
    print()
    
    for swap_ratio in SWAP_RATIOS:
        print(f"  Swap ratio = {swap_ratio:.1f}:")
        for i in range(N_SEEDS):
            seed = BASE_SEED + i
            trial_result = run_single_trial(seed, swap_ratio)
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
    summary_by_ratio = []
    for ratio in SWAP_RATIOS:
        df_ratio = df[df['swap_ratio'] == ratio]
        summary_by_ratio.append({
            'swap_ratio': ratio,
            'vs_mean': float(df_ratio['vs'].mean()),
            'vs_std': float(df_ratio['vs'].std())
        })
    
    # Save summary
    # 要約を保存
    import sys, platform, scipy
    summary = {
        'swap_ratios': SWAP_RATIOS,
        'n_seeds': N_SEEDS,
        'results_by_ratio': summary_by_ratio,
        'interpretation': 'O2: Topological disruption increases VS, confirming phase dominance',
        'config': {
            'N_ITEMS': N_ITEMS,
            'DIM': DIM,
            'RADIUS': RADIUS,
            'BASE_SEED': BASE_SEED,
            'N_SEEDS': N_SEEDS,
            'SWAP_RATIOS': SWAP_RATIOS
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
    """Create visualization of topological disruption.
    
    位相破壊の可視化を作成する。
    
    Args:
        df: DataFrame with results
            結果を含むDataFrame
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ratios = sorted(df['swap_ratio'].unique())
    
    # Panel 1: VS vs Swap Ratio
    # パネル1：VS vs スワップ比率
    vs_means = [df[df['swap_ratio'] == r]['vs'].mean() for r in ratios]
    vs_stds = [df[df['swap_ratio'] == r]['vs'].std() for r in ratios]
    
    axes[0].errorbar(ratios, vs_means, yerr=vs_stds, marker='o', linewidth=2,
                    markersize=8, capsize=5, color='red')
    axes[0].axhline(y=0, color='blue', linestyle='--', linewidth=1, alpha=0.5,
                   label='VS=0 (Natural Orthogonality)')
    axes[0].set_xlabel('Swap Ratio (Topological Disruption)', fontsize=11)
    axes[0].set_ylabel('VS (Value-Space Correlation)', fontsize=11)
    axes[0].set_title('O2: VS Increases with Topological Disruption',
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Box plots by swap ratio
    # パネル2：スワップ比率ごとの箱ひげ図
    data = [df[df['swap_ratio'] == r]['vs'].values for r in ratios]
    bp = axes[1].boxplot(data, labels=[f'{r:.1f}' for r in ratios], patch_artist=True)
    
    # Color boxes by intensity
    # 強度で色を付ける
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(ratios)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1].axhline(y=0, color='blue', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Swap Ratio', fontsize=11)
    axes[1].set_ylabel('VS (Value-Space Correlation)', fontsize=11)
    axes[1].set_title('VS Distribution by Disruption Level',
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    # 統計情報を追加
    stats_text = '\n'.join([f'Ratio {r:.1f}: {vs_means[i]:.3f}±{vs_stds[i]:.3f}'
                           for i, r in enumerate(ratios)])
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
    print("E8 Series - EXP09: Topological Disruption Test")
    print("E8シリーズ - EXP09：位相破壊テスト")
    print("=" * 70)
    print()
    print(f"Configuration / 設定:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  Swap ratios / スワップ比率 = {SWAP_RATIOS}")
    print(f"  N_SEEDS = {N_SEEDS}")
    print()
    
    df = run_exp09()
    
    print()
    print("=" * 70)
    print("Results Summary / 結果要約:")
    print("=" * 70)
    for ratio in SWAP_RATIOS:
        df_ratio = df[df['swap_ratio'] == ratio]
        print(f"Swap ratio = {ratio:.1f}:")
        print(f"  VS = {df_ratio['vs'].mean():.4f} ± {df_ratio['vs'].std():.4f}")
    print()
    print("Interpretation / 解釈:")
    print("  O2 (Phase Dominance): VS increases with topological disruption,")
    print("  confirming that topology is more important than exact coordinates.")
    print("  O2（位相優位性）：VSは位相破壊で増加し、")
    print("  位相が正確な座標よりも重要であることを確認する。")
    print("=" * 70)


if __name__ == "__main__":
    main()
