"""E8 Series - EXP10: Rotation Invariance Test

回転不変性テスト

This experiment complements EXP09 by testing whether spatial rotations
(which preserve topology) maintain VS≈0. This confirms that topology (φ)
is what matters, not absolute spatial orientation.

この実験は、位相を保存する空間回転がVS≈0を維持するかをテストすることで
EXP09を補完する。これは、絶対的な空間方向ではなく位相（φ）が重要である
ことを確認する。

Key Finding / 主要な発見:
    VS remains ≈0 across all rotation angles, confirming that topology
    is invariant to spatial transformations that preserve order.
    
    VSはすべての回転角度で≈0のままであり、位相が順序を保存する
    空間変換に対して不変であることを確認する。

References / 参考文献:
    E8b (2025-138): O2 Phase Dominance Law
    E8c (2025-139): Topological invariance
    
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
ROTATION_ANGLES = [0, 30, 60, 90, 120, 180]  # Rotation angles in degrees / 回転角度（度）
OUTPUT_DIR = Path("outputs/exp10")  # Output directory / 出力ディレクトリ


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


def rotate_coordinates(coords: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate coordinates by specified angle.
    
    指定された角度で座標を回転する。
    
    This transformation preserves topology (circular order) while
    changing absolute spatial orientation.
    
    この変換は、絶対的な空間方向を変更しながら
    位相（円周順序）を保存する。
    
    Args:
        coords: Original coordinates (N_ITEMS, 2)
               元の座標 (N_ITEMS, 2)
        angle_degrees: Rotation angle in degrees
                      回転角度（度）
                      
    Returns:
        Rotated coordinates (N_ITEMS, 2)
        回転された座標 (N_ITEMS, 2)
    """
    angle_rad = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    rotated = coords @ rotation_matrix.T
    return rotated


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


def run_single_trial(seed: int, angle: float) -> Dict[str, float]:
    """Run a single trial with rotation.
    
    回転を伴う単一試行を実行する。
    
    Args:
        seed: Random seed
              乱数シード
        angle: Rotation angle in degrees
              回転角度（度）
              
    Returns:
        Dictionary with results
        結果を含む辞書
    """
    # Generate with independent seeds
    # 独立したシードで生成
    seed_A = seed
    seed_spatial = seed + 10000
    
    A = generate_A_axis(seed_A)
    D_semantic = squareform(pdist(A, metric='correlation'))
    
    # Circle arrangement with rotation
    # 回転を伴う円周配置
    coords_original, ordering = arrange_circle(seed_spatial)
    coords_rotated = rotate_coordinates(coords_original, angle)
    D_spatial = squareform(pdist(coords_rotated, metric='euclidean'))
    
    # Compute VS
    # VSを計算
    vs = compute_vs(D_semantic, D_spatial)
    
    return {
        'seed': seed,
        'angle': angle,
        'vs': vs
    }


def run_exp10() -> pd.DataFrame:
    """Run complete EXP10 with rotation angle sweep.
    
    回転角度掃引を伴うEXP10の完全実行を行う。
    
    Returns:
        DataFrame with results
        結果を含むDataFrame
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    total = len(ROTATION_ANGLES) * N_SEEDS
    
    print(f"  Running {len(ROTATION_ANGLES)} angles × {N_SEEDS} seeds = {total} trials")
    print(f"  {len(ROTATION_ANGLES)}角度 × {N_SEEDS}シード = {total}試行を実行")
    print()
    
    for angle in ROTATION_ANGLES:
        print(f"  Rotation angle = {angle}°:")
        for i in range(N_SEEDS):
            seed = BASE_SEED + i
            trial_result = run_single_trial(seed, angle)
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
    summary_by_angle = []
    for angle in ROTATION_ANGLES:
        df_angle = df[df['angle'] == angle]
        summary_by_angle.append({
            'angle': angle,
            'vs_mean': float(df_angle['vs'].mean()),
            'vs_std': float(df_angle['vs'].std())
        })
    
    # Save summary
    # 要約を保存
    import sys, platform, scipy
    summary = {
        'rotation_angles': ROTATION_ANGLES,
        'n_seeds': N_SEEDS,
        'results_by_angle': summary_by_angle,
        'interpretation': 'O2: VS≈0 invariant to rotation, confirming topological preservation',
        'config': {
            'N_ITEMS': N_ITEMS,
            'DIM': DIM,
            'RADIUS': RADIUS,
            'BASE_SEED': BASE_SEED,
            'N_SEEDS': N_SEEDS,
            'ROTATION_ANGLES': ROTATION_ANGLES
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
    """Create visualization of rotation invariance.
    
    回転不変性の可視化を作成する。
    
    Args:
        df: DataFrame with results
            結果を含むDataFrame
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    angles = sorted(df['angle'].unique())
    
    # Panel 1: VS vs Rotation Angle
    # パネル1：VS vs 回転角度
    vs_means = [df[df['angle'] == a]['vs'].mean() for a in angles]
    vs_stds = [df[df['angle'] == a]['vs'].std() for a in angles]
    
    axes[0].errorbar(angles, vs_means, yerr=vs_stds, marker='o', linewidth=2,
                    markersize=8, capsize=5, color='blue')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5,
                   label='VS=0 (Natural Orthogonality)')
    axes[0].set_xlabel('Rotation Angle (degrees) / 回転角度（度）', fontsize=11)
    axes[0].set_ylabel('VS (Value-Space Correlation)\nVS（意味-空間相関）', fontsize=11)
    axes[0].set_title('O2: VS≈0 Invariant to Rotation\nO2：回転に対してVS≈0が不変',
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Box plots by angle
    # パネル2：角度ごとの箱ひげ図
    data = [df[df['angle'] == a]['vs'].values for a in angles]
    bp = axes[1].boxplot(data, labels=[f'{a}°' for a in angles], patch_artist=True)
    
    # Color all boxes the same (rotation preserves topology)
    # すべてのボックスを同じ色に（回転は位相を保存）
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Rotation Angle / 回転角度', fontsize=11)
    axes[1].set_ylabel('VS (Value-Space Correlation)\nVS（意味-空間相関）', fontsize=11)
    axes[1].set_title('VS Distribution Across Rotations\n回転にわたるVS分布',
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    # 統計情報を追加
    stats_text = '\n'.join([f'{a}°: {vs_means[i]:.3f}±{vs_stds[i]:.3f}'
                           for i, a in enumerate(angles)])
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
    print("E8 Series - EXP10: Rotation Invariance Test")
    print("E8シリーズ - EXP10：回転不変性テスト")
    print("=" * 70)
    print()
    print(f"Configuration / 設定:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  Rotation angles / 回転角度 = {ROTATION_ANGLES}")
    print(f"  N_SEEDS = {N_SEEDS}")
    print()
    
    df = run_exp10()
    
    print()
    print("=" * 70)
    print("Results Summary / 結果要約:")
    print("=" * 70)
    for angle in ROTATION_ANGLES:
        df_angle = df[df['angle'] == angle]
        print(f"Rotation = {angle}°:")
        print(f"  VS = {df_angle['vs'].mean():.4f} ± {df_angle['vs'].std():.4f}")
    print()
    print("Interpretation / 解釈:")
    print("  O2 (Phase Dominance): VS remains ≈0 across all rotations,")
    print("  confirming that topology is invariant to spatial transformations.")
    print("  O2（位相優位性）：VSはすべての回転で≈0のままであり、")
    print("  位相が空間変換に対して不変であることを確認する。")
    print("=" * 70)


if __name__ == "__main__":
    main()
