"""E8 Series - EXP12: Semantic Noise Test (Meaning Stress)

意味ノイズテスト（意味ストレス）

This experiment complements EXP11 by testing what happens when semantic
structure (meaning) is disrupted while spatial structure is preserved.
This provides evidence for O3 (Stress Tolerance) from the opposite direction:
meaning disruption does not automatically cause spatial confusion.

この実験は、空間構造を保存しながら意味構造（意味）を破壊したときに
何が起こるかをテストすることでEXP11を補完する。
これは、反対方向からO3（ストレス耐性）の証拠を提供する：
意味の破壊は自動的に空間的混乱を引き起こさない。

Key Finding / 主要な発見:
    Semantic noise disrupts meaning but VS remains ≈0.
    This supports O3: meaning disruption ≠ spatial confusion.
    
    意味ノイズは意味を破壊するが、VSは≈0のままである。
    これはO3を支持する：意味の破壊 ≠ 空間的混乱。

References / 参考文献:
    E8b (2025-138): O3 Stress Tolerance Law
    E8c (2025-139): Bidirectional independence
    
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
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.5, 1.0]  # Semantic noise levels / 意味ノイズレベル
OUTPUT_DIR = Path("outputs/exp12")  # Output directory / 出力ディレクトリ


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


def add_semantic_noise(A: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Add semantic noise by mixing with random embeddings.
    
    ランダムな埋め込みと混合することで意味ノイズを追加する。
    
    This disrupts semantic relationships while maintaining the embedding
    structure. Higher noise levels increase semantic confusion.
    
    これは、埋め込み構造を維持しながら意味関係を破壊する。
    高いノイズレベルは意味的混乱を増加させる。
    
    Args:
        A: Original A-axis
           元のA軸
        noise_level: Mixing ratio with random embeddings [0, 1]
                    ランダム埋め込みとの混合比率 [0, 1]
        seed: Random seed for noise
              ノイズのための乱数シード
              
    Returns:
        Noisy A-axis (re-normalized)
        ノイズを含むA軸（再正規化済み）
    """
    rng = np.random.default_rng(seed)
    
    # Generate random embeddings
    # ランダムな埋め込みを生成
    A_random = rng.standard_normal(A.shape)
    A_random = A_random / (np.linalg.norm(A_random, axis=1, keepdims=True) + 1e-12)
    
    # Mix original and random: A_noisy = (1-λ)*A + λ*A_random
    # 元とランダムを混合：A_noisy = (1-λ)*A + λ*A_random
    A_noisy = (1 - noise_level) * A + noise_level * A_random
    
    # Re-normalize
    # 再正規化
    norms = np.linalg.norm(A_noisy, axis=1, keepdims=True)
    A_noisy = A_noisy / (norms + 1e-12)
    
    return A_noisy


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


def compute_semantic_similarity(A_original: np.ndarray, A_noisy: np.ndarray) -> float:
    """Compute semantic similarity between original and noisy A-axis.
    
    元のA軸とノイズを含むA軸の意味類似度を計算する。
    
    Uses Spearman correlation between distance matrices.
    距離行列間のスピアマン相関を使用する。
    
    Args:
        A_original: Original A-axis
                   元のA軸
        A_noisy: Noisy A-axis
                ノイズを含むA軸
                
    Returns:
        Spearman correlation in [-1, 1]
        範囲[-1, 1]のスピアマン相関
    """
    d_orig = pdist(A_original, metric='correlation')
    d_noisy = pdist(A_noisy, metric='correlation')
    sim, _ = spearmanr(d_orig, d_noisy)
    return float(sim)


def run_single_trial(seed: int, noise_level: float) -> Dict[str, float]:
    """Run a single trial with semantic noise.
    
    意味ノイズを伴う単一試行を実行する。
    
    Args:
        seed: Random seed
              乱数シード
        noise_level: Semantic noise level
                    意味ノイズレベル
                    
    Returns:
        Dictionary with results
        結果を含む辞書
    """
    # Generate with independent seeds
    # 独立したシードで生成
    seed_A = seed
    seed_noise = seed + 10000
    seed_spatial = seed + 20000
    
    # Original A-axis
    # 元のA軸
    A_original = generate_A_axis(seed_A)
    
    # Add semantic noise
    # 意味ノイズを追加
    A_noisy = add_semantic_noise(A_original, noise_level, seed_noise)
    
    # Compute semantic similarity
    # 意味類似度を計算
    sem_sim = compute_semantic_similarity(A_original, A_noisy)
    
    # Spatial arrangement (fixed structure)
    # 空間配置（固定構造）
    coords, ordering = arrange_circle(seed_spatial)
    D_spatial = squareform(pdist(coords, metric='euclidean'))
    
    # Compute VS with noisy semantics
    # ノイズを含む意味でVSを計算
    D_semantic = squareform(pdist(A_noisy, metric='correlation'))
    vs = compute_vs(D_semantic, D_spatial)
    
    return {
        'seed': seed,
        'noise_level': noise_level,
        'vs': vs,
        'semantic_similarity': sem_sim
    }


def run_exp12() -> pd.DataFrame:
    """Run complete EXP12 with semantic noise sweep.
    
    意味ノイズ掃引を伴うEXP12の完全実行を行う。
    
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
        print(f"  Semantic noise = {noise_level:.1f}:")
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
            'vs_std': float(df_noise['vs'].std()),
            'semantic_similarity_mean': float(df_noise['semantic_similarity'].mean()),
            'semantic_similarity_std': float(df_noise['semantic_similarity'].std())
        })
    
    # Save summary
    # 要約を保存
    import sys, platform, scipy
    summary = {
        'noise_levels': NOISE_LEVELS,
        'n_seeds': N_SEEDS,
        'results_by_noise': summary_by_noise,
        'interpretation': 'O3: Semantic disruption does not cause spatial confusion (VS≈0)',
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
    """Create visualization of semantic stress test.
    
    意味ストレステストの可視化を作成する。
    
    Args:
        df: DataFrame with results
            結果を含むDataFrame
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    noise_vals = sorted(df['noise_level'].unique())
    
    # Panel 1: VS vs Semantic Noise
    # パネル1：VS vs 意味ノイズ
    vs_means = [df[df['noise_level'] == n]['vs'].mean() for n in noise_vals]
    vs_stds = [df[df['noise_level'] == n]['vs'].std() for n in noise_vals]
    
    axes[0].errorbar(noise_vals, vs_means, yerr=vs_stds, marker='o', linewidth=2,
                    markersize=8, capsize=5, color='purple', label='VS')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('Semantic Noise Level\n意味ノイズレベル', fontsize=11)
    axes[0].set_ylabel('VS (Value-Space Correlation)\nVS（意味-空間相関）', fontsize=11)
    axes[0].set_title('O3: VS≈0 Despite Semantic Stress\nO3：意味ストレス下でもVS≈0',
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Semantic Similarity vs Noise
    # パネル2：意味類似度 vs ノイズ
    sem_sim_means = [df[df['noise_level'] == n]['semantic_similarity'].mean() for n in noise_vals]
    sem_sim_stds = [df[df['noise_level'] == n]['semantic_similarity'].std() for n in noise_vals]
    
    axes[1].errorbar(noise_vals, sem_sim_means, yerr=sem_sim_stds, marker='^',
                    linewidth=2, markersize=8, capsize=5, color='orange',
                    label='Semantic Similarity')
    axes[1].axhline(y=1.0, color='blue', linestyle='--', linewidth=1, alpha=0.5,
                   label='Perfect similarity')
    axes[1].set_xlabel('Semantic Noise Level\n意味ノイズレベル', fontsize=11)
    axes[1].set_ylabel('Semantic Similarity\n意味類似度', fontsize=11)
    axes[1].set_title('Meaning Degrades with Noise\n意味はノイズで劣化',
                     fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Semantic Similarity vs VS scatter
    # パネル3：意味類似度 vs VS散布図
    colors_map = {n: plt.cm.viridis(i/len(noise_vals)) for i, n in enumerate(noise_vals)}
    
    for noise in noise_vals:
        df_noise = df[df['noise_level'] == noise]
        axes[2].scatter(df_noise['semantic_similarity'], df_noise['vs'],
                       alpha=0.5, s=30, c=[colors_map[noise]],
                       label=f'Noise={noise:.1f}')
    
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[2].set_xlabel('Semantic Similarity\n意味類似度', fontsize=11)
    axes[2].set_ylabel('VS (Value-Space Correlation)\nVS（意味-空間相関）', fontsize=11)
    axes[2].set_title('O3: Low Semantic Similarity ≠ High VS\nO3：低意味類似度 ≠ 高VS',
                     fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    # Add interpretation
    # 解釈を追加
    interpretation = (
        f'Noise 0.0: VS={vs_means[0]:.3f}, Sim={sem_sim_means[0]:.3f}\n'
        f'Noise 1.0: VS={vs_means[-1]:.3f}, Sim={sem_sim_means[-1]:.3f}\n'
        f'ΔVS = {vs_means[-1] - vs_means[0]:.3f}\n'
        f'ΔSim = {sem_sim_means[-1] - sem_sim_means[0]:.3f}\n\n'
        'Meaning degrades but structure remains orthogonal'
    )
    fig.text(0.5, 0.02, interpretation, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    # 図を保存
    fig_path = OUTPUT_DIR / "visualization.png"
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved to / 可視化を保存: {fig_path}")


def main():
    """Main execution function.
    
    メイン実行関数。
    """
    print("=" * 70)
    print("E8 Series - EXP12: Semantic Noise Test")
    print("E8シリーズ - EXP12：意味ノイズテスト")
    print("=" * 70)
    print()
    print(f"Configuration / 設定:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  Semantic noise levels / 意味ノイズレベル = {NOISE_LEVELS}")
    print(f"  N_SEEDS = {N_SEEDS}")
    print()
    
    df = run_exp12()
    
    print()
    print("=" * 70)
    print("Results Summary / 結果要約:")
    print("=" * 70)
    for noise in NOISE_LEVELS:
        df_noise = df[df['noise_level'] == noise]
        print(f"Semantic noise = {noise:.1f}:")
        print(f"  VS = {df_noise['vs'].mean():.4f} ± {df_noise['vs'].std():.4f}")
        print(f"  Sem Sim = {df_noise['semantic_similarity'].mean():.4f}")
    print()
    print("Interpretation / 解釈:")
    print("  O3 (Stress Tolerance): Semantic disruption does not cause")
    print("  spatial confusion (VS remains ≈0).")
    print("  O3（ストレス耐性）：意味の破壊は空間的混乱を引き起こさない")
    print("  （VSは≈0のまま）。")
    print()
    print("  This complements EXP11, showing bidirectional independence:")
    print("  structure ⊥ meaning regardless of which is disrupted.")
    print("  これはEXP11を補完し、双方向の独立性を示す：")
    print("  どちらが破壊されても、構造 ⊥ 意味。")
    print("=" * 70)


if __name__ == "__main__":
    main()
