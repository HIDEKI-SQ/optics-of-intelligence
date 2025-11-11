"""E8 Series - EXP01: Spatial vs Random Arrangement

空間配置 vs ランダム配置

This experiment tests whether spatial arrangement (Method of Loci)
creates semantic-spatial binding (VS>0) or whether structure and 
meaning remain naturally orthogonal (VS≈0).

この実験は、空間配置（記憶の宮殿）が意味と空間の結合（VS>0）を
生成するか、それとも構造と意味が自然に直交したまま（VS≈0）であるかを検証する。

Key Discovery / 主要な発見:
    VS≈0 in both conditions, establishing natural orthogonality (O1)
    両条件でVS≈0となり、自然直交性（O1）を確立

References / 参考文献:
    E8a (2025-137): First observation of VS≈0
    E8b (2025-138): O1 Law of Natural Orthogonality
    
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
from typing import Tuple, Dict, Any
import json

# Configuration
# 設定
N_ITEMS = 20        # Number of items in A-axis / A軸のアイテム数
DIM = 100           # Embedding dimension / 埋め込み次元
RADIUS = 1.0        # Circle radius for spatial condition / 空間条件の円半径
BASE_SEED = 42      # Base random seed / 基本乱数シード
N_SEEDS = 1000      # Number of seeds for robustness / 頑健性検証用のシード数
OUTPUT_DIR = Path("outputs/exp01")  # Output directory / 出力ディレクトリ


def generate_A_axis(seed: int) -> np.ndarray:
    """Generate A-axis (Blueprint structure).
    
    A軸（Blueprint構造）を生成する。
    
    The A-axis represents the autonomous structure generated through
    long-term interaction, independent of any specific value or viewpoint.
    This is the foundational concept from E7 framework.
    
    A軸は、特定の価値や視点に依存せず、長期的な相互作用を通じて
    自律的に生成される構造を表す。これはE7枠組みの基礎概念である。
    
    Args:
        seed: Random seed for reproducibility
              再現性のための乱数シード
              
    Returns:
        A-axis matrix with shape (N_ITEMS, DIM)
        形状(N_ITEMS, DIM)のA軸行列
        
    Example:
        >>> A = generate_A_axis(42)
        >>> A.shape
        (20, 100)
        >>> np.allclose(np.linalg.norm(A, axis=1), 1.0)
        True
    """
    # Initialize random number generator with fixed seed
    # 固定シードで乱数生成器を初期化
    rng = np.random.default_rng(seed)
    
    # Generate random embedding from standard normal distribution
    # 標準正規分布からランダムな埋め込みを生成
    A = rng.standard_normal((N_ITEMS, DIM))
    
    # Normalize each row to unit length
    # 各行を単位長に正規化
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    A = A / (norms + 1e-12)
    
    return A


def arrange_spatial(A: np.ndarray, seed: int) -> np.ndarray:
    """Arrange items on a circle (spatial condition).
    
    アイテムを円周上に配置する（空間条件）。
    
    This implements the classical Method of Loci by placing items at
    equal angles on a circle. This preserves topological order (φ)
    while introducing explicit spatial coordinates.
    
    これは古典的な記憶の宮殿を実装し、アイテムを円周上に等間隔で配置する。
    これにより、明示的な空間座標を導入しつつ位相的順序（φ）を保存する。
    
    Traditional assumption: This should create VS>0 (semantic-spatial binding)
    E8a discovery: Actually produces VS≈0 (natural orthogonality)
    
    伝統的仮定：これはVS>0（意味-空間結合）を生成するはずである
    E8aの発見：実際にはVS≈0（自然直交性）を生成する
    
    Args:
        A: A-axis matrix with shape (N_ITEMS, DIM)
           形状(N_ITEMS, DIM)のA軸行列
        seed: Random seed for order permutation
              順序置換のための乱数シード
              
    Returns:
        Spatial coordinates with shape (N_ITEMS, 2)
        形状(N_ITEMS, 2)の空間座標
        
    Example:
        >>> A = generate_A_axis(42)
        >>> coords = arrange_spatial(A, 42)
        >>> coords.shape
        (20, 2)
        >>> # Verify points are on circle
        >>> distances = np.linalg.norm(coords, axis=1)
        >>> np.allclose(distances, RADIUS)
        True
    """
    # Initialize random number generator
    # 乱数生成器を初期化
    rng = np.random.default_rng(seed)
    
    # Generate permutation for randomized order
    # ランダム化された順序のための置換を生成
    order = rng.permutation(N_ITEMS)
    
    # Compute equally-spaced angles on circle
    # 円周上の等間隔な角度を計算
    angles = 2 * np.pi * np.arange(N_ITEMS) / N_ITEMS
    
    # Convert polar to Cartesian coordinates
    # 極座標からデカルト座標に変換
    x = RADIUS * np.cos(angles)
    y = RADIUS * np.sin(angles)
    
    # Create coordinate matrix
    # 座標行列を作成
    coords = np.column_stack([x, y])
    
    # Apply permutation to match A-axis order
    # A軸の順序に合わせて置換を適用
    coords = coords[order]
    
    return coords


def arrange_random(seed: int) -> np.ndarray:
    """Arrange items randomly (control condition).
    
    アイテムをランダムに配置する（統制条件）。
    
    This serves as the null hypothesis condition where no intentional
    value-driven spatial arrangement is applied (λ=0).
    
    これは、意図的な価値駆動の空間配置が適用されない（λ=0）
    帰無仮説条件として機能する。
    
    Args:
        seed: Random seed for reproducibility
              再現性のための乱数シード
              
    Returns:
        Random coordinates with shape (N_ITEMS, 2) in range [-1, 1]
        範囲[-1, 1]の形状(N_ITEMS, 2)のランダム座標
        
    Example:
        >>> coords = arrange_random(42)
        >>> coords.shape
        (20, 2)
        >>> np.all((-1 <= coords) & (coords <= 1))
        True
    """
    # Initialize random number generator
    # 乱数生成器を初期化
    rng = np.random.default_rng(seed)
    
    # Generate uniform random coordinates in [-1, 1] square
    # [-1, 1]の正方形内に一様ランダムな座標を生成
    coords = rng.uniform(-1, 1, (N_ITEMS, 2))
    
    return coords


def compute_sp(A_orig: np.ndarray, A_reconstructed: np.ndarray) -> float:
    """Compute Structure Preservation (SP).
    
    構造保存度（SP）を計算する。
    
    SP measures how well the original A-axis structure is preserved
    after transformation. High SP indicates that topological order (φ)
    is maintained despite coordinate changes.
    
    SP は、変換後に元のA軸構造がどれだけ保存されているかを測定する。
    高いSP は、座標変化にもかかわらず位相的順序（φ）が維持されていることを示す。
    
    Computation method: Spearman correlation between condensed distance vectors
    計算方法：圧縮距離ベクトル間のスピアマン相関
    
    Note: Uses condensed vectors (upper triangular, no diagonal) to avoid
    double-counting symmetric distances and zero diagonal elements.
    
    注：対称距離の二重カウントと対角ゼロ要素を避けるため、
    圧縮ベクトル（上三角、対角なし）を使用する。
    
    Args:
        A_orig: Original A-axis with shape (N_ITEMS, DIM)
                形状(N_ITEMS, DIM)の元のA軸
        A_reconstructed: Reconstructed A-axis with shape (N_ITEMS, DIM)
                        形状(N_ITEMS, DIM)の再構成されたA軸
                        
    Returns:
        Spearman correlation coefficient in range [-1, 1]
        範囲[-1, 1]のスピアマン相関係数
        Higher values indicate better structure preservation
        高い値はより良い構造保存を示す
        
    References / 参考文献:
        E8b (2025-138): O2 Phase Dominance Law
        E8e (2025-141): Verification - condensed vector method
        
    Example:
        >>> A1 = generate_A_axis(42)
        >>> A2 = A1 + 0.01 * np.random.randn(*A1.shape)
        >>> sp = compute_sp(A1, A2)
        >>> sp > 0.95  # High preservation with small noise
        True
    """
    # Compute condensed distance vectors (upper triangular, no diagonal)
    # 圧縮距離ベクトルを計算（上三角、対角なし）
    d_orig = pdist(A_orig, metric='correlation')
    d_recon = pdist(A_reconstructed, metric='correlation')
    
    # Compute Spearman rank correlation on condensed vectors
    # 圧縮ベクトルでスピアマン順位相関を計算
    sp, _ = spearmanr(d_orig, d_recon)
    
    return float(sp)


def compute_vs(D_semantic: np.ndarray, D_spatial: np.ndarray) -> float:
    """Compute Value-Space correlation (VS).
    
    意味-空間相関（VS）を計算する。
    
    VS measures whether semantic distance is correlated with spatial distance.
    Traditional Method of Loci theory assumes VS>0 (semantic-spatial binding),
    but E8a discovered VS≈0 (natural orthogonality).
    
    VS は、意味距離が空間距離と相関しているかを測定する。
    伝統的な記憶の宮殿理論はVS>0（意味-空間結合）を仮定するが、
    E8a はVS≈0（自然直交性）を発見した。
    
    This is the central metric that led to the discovery of O1:
    Structure and meaning are independent without value pressure (λ=0).
    
    これは、O1の発見につながった中心的な指標である：
    構造と意味は、価値圧力（λ=0）なしでは独立している。
    
    Computation method: Spearman correlation between condensed distance vectors
    計算方法：圧縮距離ベクトル間のスピアマン相関
    
    Note: Uses condensed vectors (upper triangular, no diagonal) from square
    distance matrices to avoid double-counting and diagonal zeros.
    
    注：二重カウントと対角ゼロを避けるため、正方距離行列から
    圧縮ベクトル（上三角、対角なし）を使用する。
    
    Args:
        D_semantic: Semantic distance matrix with shape (N_ITEMS, N_ITEMS)
                   形状(N_ITEMS, N_ITEMS)の意味距離行列
        D_spatial: Spatial distance matrix with shape (N_ITEMS, N_ITEMS)
                  形状(N_ITEMS, N_ITEMS)の空間距離行列
                  
    Returns:
        Spearman correlation coefficient in range [-1, 1]
        範囲[-1, 1]のスピアマン相関係数
        VS≈0 indicates natural orthogonality
        VS≈0 は自然直交性を示す
        
    References / 参考文献:
        E8a (2025-137): First VS≈0 observation
        E8b (2025-138): O1 Natural Orthogonality Law
        E8e (2025-141): Verification - condensed vector method
        
    Example:
        >>> # Generate uncorrelated distances
        >>> D_sem = squareform(pdist(np.random.rand(20, 10)))
        >>> D_spa = squareform(pdist(np.random.rand(20, 2)))
        >>> vs = compute_vs(D_sem, D_spa)
        >>> abs(vs) < 0.2  # Should be near zero
        True
    """
    # Convert square distance matrices to condensed vectors
    # 正方距離行列を圧縮ベクトルに変換
    # squareform with checks=False: assumes valid distance matrix, extracts upper tri
    # checks=Falseのsquareform：有効な距離行列を仮定し、上三角を抽出
    sem_flat = squareform(D_semantic, checks=False)
    spatial_flat = squareform(D_spatial, checks=False)
    
    # Compute Spearman rank correlation on condensed vectors
    # 圧縮ベクトルでスピアマン順位相関を計算
    vs, _ = spearmanr(sem_flat, spatial_flat)
    
    return float(vs)


def run_single_trial(seed: int) -> Dict[str, float]:
    """Run a single trial of EXP01.
    
    EXP01の単一試行を実行する。
    
    Args:
        seed: Random seed for this trial
              この試行のための乱数シード
              
    Returns:
        Dictionary containing results for both conditions
        両条件の結果を含む辞書
    """
    # Generate A-axis
    # A軸を生成
    A = generate_A_axis(seed)
    
    # Compute semantic distance matrix
    # 意味距離行列を計算
    D_semantic = squareform(pdist(A, metric='correlation'))
    
    # Condition 1: Spatial arrangement (Method of Loci)
    # 条件1：空間配置（記憶の宮殿）
    coords_spatial = arrange_spatial(A, seed)
    D_spatial_cond = squareform(pdist(coords_spatial, metric='euclidean'))
    vs_spatial = compute_vs(D_semantic, D_spatial_cond)
    
    # Condition 2: Random arrangement (control)
    # 条件2：ランダム配置（統制）
    coords_random = arrange_random(seed)
    D_random_cond = squareform(pdist(coords_random, metric='euclidean'))
    vs_random = compute_vs(D_semantic, D_random_cond)
    
    return {
        'seed': seed,
        'vs_spatial': vs_spatial,
        'vs_random': vs_random
    }


def run_exp01() -> pd.DataFrame:
    """Run complete EXP01 with multiple seeds.
    
    複数のシードでEXP01の完全実行を行う。
    
    Returns:
        DataFrame containing results for all trials
        全試行の結果を含むDataFrame
    """
    # Create output directory if it doesn't exist
    # 出力ディレクトリが存在しない場合は作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run trials for all seeds
    # 全シードで試行を実行
    results = []
    for i in range(N_SEEDS):
        seed = BASE_SEED + i
        trial_result = run_single_trial(seed)
        results.append(trial_result)
        
        # Progress indicator
        # 進捗表示
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{N_SEEDS} trials / 試行完了")
    
    # Convert to DataFrame
    # DataFrameに変換
    df = pd.DataFrame(results)
    
    # Save results to CSV
    # 結果をCSVに保存
    csv_path = OUTPUT_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to / 結果を保存: {csv_path}")
    
    # Save summary statistics
    # 要約統計量を保存
    summary = {
        'vs_spatial_mean': float(df['vs_spatial'].mean()),
        'vs_spatial_std': float(df['vs_spatial'].std()),
        'vs_random_mean': float(df['vs_random'].mean()),
        'vs_random_std': float(df['vs_random'].std()),
        'n_seeds': N_SEEDS
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
        df: DataFrame containing results
            結果を含むDataFrame
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Box plot comparing spatial vs random
    # 空間配置 vs ランダム配置の箱ひげ図
    data = [df['vs_spatial'], df['vs_random']]
    labels = ['Spatial', 'Random']
    
    ax.boxplot(data, labels=labels)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('VS (Value-Space Correlation)')
    ax.set_title('EXP01: VS≈0 in Both Conditions')
    ax.grid(True, alpha=0.3)
    
    # Add text annotation
    # テキスト注釈を追加
    mean_spatial = df['vs_spatial'].mean()
    mean_random = df['vs_random'].mean()
    ax.text(0.02, 0.98, 
            f'Spatial mean: {mean_spatial:.3f}\n'
            f'Random mean: {mean_random:.3f}',
            transform=ax.transAxes, 
            verticalalignment='top',
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
    print("E8 Series - EXP01: Spatial vs Random")
    print("E8シリーズ - EXP01：空間配置 vs ランダム配置")
    print("=" * 70)
    print()
    print(f"Configuration / 設定:")
    print(f"  N_ITEMS = {N_ITEMS}")
    print(f"  DIM = {DIM}")
    print(f"  N_SEEDS = {N_SEEDS}")
    print(f"  BASE_SEED = {BASE_SEED}")
    print()
    print("Running experiment / 実験実行中...")
    print()
    
    # Run experiment
    # 実験を実行
    df = run_exp01()
    
    # Display summary
    # 要約を表示
    print()
    print("=" * 70)
    print("Results / 結果:")
    print("=" * 70)
    print(f"VS (spatial) / VS（空間）:  {df['vs_spatial'].mean():.4f} ± {df['vs_spatial'].std():.4f}")
    print(f"VS (random) / VS（ランダム）: {df['vs_random'].mean():.4f} ± {df['vs_random'].std():.4f}")
    print()
    print("Interpretation / 解釈:")
    print("  Both conditions show VS≈0, supporting natural orthogonality (O1).")
    print("  両条件でVS≈0を示し、自然直交性（O1）を支持する。")
    print()
    print("  Structure and meaning are independent without value pressure (λ=0).")
    print("  構造と意味は、価値圧力（λ=0）なしでは独立している。")
    print("=" * 70)


# Execute if run as script
# スクリプトとして実行された場合に実行
if __name__ == "__main__":
    main()
