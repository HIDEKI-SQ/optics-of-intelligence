"""E8 Series - EXP Beta: Initial Exploration

初期探索実験

This is the initial exploratory experiment that preceded EXP00.
While this experiment revealed interesting patterns, methodological
concerns led to a redesign as EXP00.

これはEXP00に先行する初期探索実験である。
この実験は興味深いパターンを明らかにしたが、方法論的な懸念により
EXP00として再設計された。

Historical Note / 歴史的注記:
    This experiment is preserved for transparency and completeness.
    For rigorous analysis, refer to EXP00 and subsequent experiments.
    
    この実験は透明性と完全性のために保存されている。
    厳密な分析については、EXP00および後続の実験を参照されたい。

References / 参考文献:
    E8a (2025-137): Initial observations
    
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
N_ITEMS = 20        # Number of items in A-axis / A軸のアイテム数
DIM = 100           # Embedding dimension / 埋め込み次元
BASE_SEED = 42      # Base random seed / 基本乱数シード
N_SEEDS = 1000      # Number of seeds for robustness / 頑健性検証用のシード数
OUTPUT_DIR = Path("outputs/exp_beta")  # Output directory / 出力ディレクトリ


def generate_A_axis(seed: int) -> np.ndarray:
    """Generate A-axis (Blueprint structure).
    
    A軸（Blueprint構造）を生成する。
    
    Args:
        seed: Random seed for reproducibility
              再現性のための乱数シード
              
    Returns:
        A-axis matrix with shape (N_ITEMS, DIM)
        形状(N_ITEMS, DIM)のA軸行列
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


def arrange_random(seed: int) -> np.ndarray:
    """Arrange items randomly in 2D space.
    
    2D空間にアイテムをランダムに配置する。
    
    Args:
        seed: Random seed for reproducibility
              再現性のための乱数シード
              
    Returns:
        Random coordinates with shape (N_ITEMS, 2) in range [-1, 1]
        範囲[-1, 1]の形状(N_ITEMS, 2)のランダム座標
    """
    # Initialize random number generator
    # 乱数生成器を初期化
    rng = np.random.default_rng(seed)
    
    # Generate uniform random coordinates in [-1, 1] square
    # [-1, 1]の正方形内に一様ランダムな座標を生成
    coords = rng.uniform(-1, 1, (N_ITEMS, 2))
    
    return coords


def compute_vs(D_semantic: np.ndarray, D_spatial: np.ndarray) -> float:
    """Compute Value-Space correlation (VS).
    
    意味-空間相関（VS）を計算する。
    
    Uses condensed vectors to avoid double-counting and diagonal zeros.
    圧縮ベクトルを使用して二重カウントと対角ゼロを回避する。
    
    Args:
        D_semantic: Semantic distance matrix
                   意味距離行列
        D_spatial: Spatial distance matrix
                  空間距離行列
                  
    Returns:
        Spearman correlation coefficient
        スピアマン相関係数
    """
    # Convert to condensed vectors
    # 圧縮ベクトルに変換
    sem_flat = squareform(D_semantic, checks=False)
    spatial_flat = squareform(D_spatial, checks=False)
    
    # Compute Spearman correlation
    # スピアマン相関を計算
    vs, _ = spearmanr(sem_flat, spatial_flat)
    
    return float(vs)


def run_single_trial(seed: int) -> Dict[str, float]:
    """Run a single trial of EXP Beta.
    
    EXP Betaの単一試行を実行する。
    
    Args:
        seed: Random seed for this trial
              この試行のための乱数シード
              
    Returns:
        Dictionary containing VS result
        VS結果を含む辞書
    """
    # Generate A-axis
    # A軸を生成
    A = generate_A_axis(seed)
    
    # Compute semantic distance matrix
    # 意味距離行列を計算
    D_semantic = squareform(pdist(A, metric='correlation'))
    
    # Random arrangement
    # ランダム配置
    coords = arrange_random(seed)
    D_spatial = squareform(pdist(coords, metric='euclidean'))
    
    # Compute VS
    # VSを計算
    vs = compute_vs(D_semantic, D_spatial)
    
    return {
        'seed': seed,
        'vs': vs
    }


def run_exp_beta() -> pd.DataFrame:
    """Run complete EXP Beta with multiple seeds.
    
    複数のシードでEXP Betaの完全実行を行う。
    
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
    import sys
    import platform
    import scipy
    
    summary = {
        'vs_mean': float(df['vs'].mean()),
        'vs_std': float(df['vs'].std()),
        'vs_median': float(df['vs'].median()),
        'n_seeds': N_SEEDS,
        'note': 'Initial exploratory experiment. See EXP00 for refined methodology.',
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
        df: DataFrame containing results
            結果を含むDataFrame
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: Histogram
    # 左パネル：ヒストグラム
    ax1.hist(df['vs'], bins=15, edgecolor='black', alpha=0.7, color='lightblue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='VS=0')
    ax1.axvline(x=df['vs'].mean(), color='blue', linestyle='-', linewidth=2, 
                label=f'Mean={df["vs"].mean():.3f}')
    ax1.set_xlabel('VS (Value-Space Correlation)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('EXP Beta: VS Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Box plot
    # 右パネル：箱ひげ図
    ax2.boxplot([df['vs']], labels=['Beta\nベータ'])
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('VS (Value-Space Correlation)')
    ax2.set_title('EXP Beta: Exploratory Results')
    ax2.grid(True, alpha=0.3)
    
    # Add note
    # 注記を追加
    note_text = (
        'Initial Exploration\n\n'
        f'Mean: {df["vs"].mean():.4f}\n'
        f'Std: {df["vs"].std():.4f}\n\n'
        'See EXP00 for refined method\n'
    )
    ax2.text(0.02, 0.98, note_text,
            transform=ax2.transAxes,
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
    print("E8 Series - EXP Beta: Initial Exploration")
    print("E8シリーズ - EXP Beta：初期探索実験")
    print("=" * 70)
    print()
    print("Note: This is the initial exploratory experiment.")
    print("      For refined methodology, see EXP00.")
    print("注記：これは初期探索実験である。")
    print("      改良された方法論についてはEXP00を参照されたい。")
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
    df = run_exp_beta()
    
    # Display summary
    # 要約を表示
    print()
    print("=" * 70)
    print("Results / 結果:")
    print("=" * 70)
    print(f"VS: {df['vs'].mean():.4f} ± {df['vs'].std():.4f}")
    print(f"Median / 中央値: {df['vs'].median():.4f}")
    print()
    print("Interpretation / 解釈:")
    print("  Initial observation of VS≈0 pattern.")
    print("  VS≈0パターンの初期観測。")
    print()
    print("  This exploratory result led to systematic investigation in EXP00-13.")
    print("  この探索的結果により、EXP00-13での系統的調査につながった。")
    print("=" * 70)


# Execute if run as script
# スクリプトとして実行された場合に実行
if __name__ == "__main__":
    main()
