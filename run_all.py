"""E8 Series - Run All Experiments

全実験実行スクリプト

This script runs all E8 experiments sequentially and generates
a comprehensive summary report.

このスクリプトは全E8実験を順次実行し、包括的な要約レポートを生成する。

Experiments executed:
実行される実験:
    - Beta: Initial exploration / 初期探索
    - EXP00-13: Complete E8 series / 完全なE8シリーズ

Author: HIDEKI
Date: 2025-11
License: MIT
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# Add src to path
# srcをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

# Import all experiments
# 全実験をインポート
from src.experiments.exp_beta import run_exp_beta
from src.experiments.exp_00 import run_exp00
from src.experiments.exp_01 import run_exp01
from src.experiments.exp_02 import run_exp02
from src.experiments.exp_03 import run_exp03
from src.experiments.exp_04 import run_exp04
from src.experiments.exp_05 import run_exp05
from src.experiments.exp_06 import run_exp06
from src.experiments.exp_07 import run_exp07
from src.experiments.exp_08 import run_exp08
from src.experiments.exp_09 import run_exp09
from src.experiments.exp_10 import run_exp10
from src.experiments.exp_11 import run_exp11
from src.experiments.exp_12 import run_exp12
from src.experiments.exp_13 import run_exp13


# Experiment registry
# 実験レジストリ
EXPERIMENTS = [
    {
        'name': 'Beta',
        'description': 'Initial Exploration / 初期探索',
        'function': run_exp_beta,
        'category': 'Exploratory'
    },
    {
        'name': 'EXP00',
        'description': 'Random Baseline / ランダムベースライン',
        'function': run_exp00,
        'category': 'O1: Natural Orthogonality'
    },
    {
        'name': 'EXP01',
        'description': 'Spatial vs Random / 空間 vs ランダム',
        'function': run_exp01,
        'category': 'O1: Natural Orthogonality'
    },
    {
        'name': 'EXP02',
        'description': 'Grid Arrangement / グリッド配置',
        'function': run_exp02,
        'category': 'O1: Natural Orthogonality'
    },
    {
        'name': 'EXP03',
        'description': 'Line Arrangement (1D) / ライン配置（1D）',
        'function': run_exp03,
        'category': 'O1: Natural Orthogonality'
    },
    {
        'name': 'EXP04',
        'description': '3D Cube Arrangement / 3D立方体配置',
        'function': run_exp04,
        'category': 'O1: Natural Orthogonality'
    },
    {
        'name': 'EXP05',
        'description': 'Independence Test (Permutation) / 独立性検定（順列）',
        'function': run_exp05,
        'category': 'O1: Natural Orthogonality'
    },
    {
        'name': 'EXP06',
        'description': 'Dimension Robustness / 次元頑健性',
        'function': run_exp06,
        'category': 'O1: Natural Orthogonality'
    },
    {
        'name': 'EXP07',
        'description': 'Sample Size Robustness / サンプルサイズ頑健性',
        'function': run_exp07,
        'category': 'O1: Natural Orthogonality'
    },
    {
        'name': 'EXP08',
        'description': 'Metric Type Robustness / 計量タイプ頑健性',
        'function': run_exp08,
        'category': 'O1: Natural Orthogonality'
    },
    {
        'name': 'EXP09',
        'description': 'Topological Disruption / 位相破壊',
        'function': run_exp09,
        'category': 'O2: Phase Dominance'
    },
    {
        'name': 'EXP10',
        'description': 'Rotation Invariance / 回転不変性',
        'function': run_exp10,
        'category': 'O2: Phase Dominance'
    },
    {
        'name': 'EXP11',
        'description': 'Structural Stress / 構造ストレス',
        'function': run_exp11,
        'category': 'O3: Stress Tolerance'
    },
    {
        'name': 'EXP12',
        'description': 'Semantic Noise / 意味ノイズ',
        'function': run_exp12,
        'category': 'O3: Stress Tolerance'
    },
    {
        'name': 'EXP13',
        'description': 'Value Gate Sweep (λ) / 価値ゲート掃引（λ）',
        'function': run_exp13,
        'category': 'O4: Value-Gated Coupling'
    }
]


def print_header():
    """Print header banner.
    
    ヘッダーバナーを表示する。
    """
    print()
    print("=" * 80)
    print("E8 SERIES - COMPLETE EXPERIMENTAL SUITE")
    print("E8シリーズ - 完全実験スイート")
    print("=" * 80)
    print()
    print("Optics of Intelligence: Structure, Value, and Meaning")
    print("知性光学：構造・価値・意味")
    print()
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"実行開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"合計実験数: {len(EXPERIMENTS)}")
    print("=" * 80)
    print()


def print_experiment_header(exp_num: int, exp_info: dict):
    """Print experiment header.
    
    実験ヘッダーを表示する。
    
    Args:
        exp_num: Experiment number (1-indexed)
                実験番号（1始まり）
        exp_info: Experiment information dictionary
                 実験情報辞書
    """
    print()
    print("-" * 80)
    print(f"[{exp_num}/{len(EXPERIMENTS)}] {exp_info['name']}: {exp_info['description']}")
    print(f"Category / カテゴリ: {exp_info['category']}")
    print("-" * 80)
    print()


def run_all_experiments(skip_completed: bool = False) -> dict:
    """Run all E8 experiments.
    
    全E8実験を実行する。
    
    Args:
        skip_completed: If True, skip experiments with existing outputs
                       Trueの場合、既存の出力がある実験をスキップ
                       
    Returns:
        Dictionary with execution summary
        実行要約を含む辞書
    """
    results = {
        'start_time': datetime.now().isoformat(),
        'experiments': [],
        'total_duration': 0,
        'success_count': 0,
        'failure_count': 0,
        'skipped_count': 0
    }
    
    overall_start = time.time()
    
    for i, exp_info in enumerate(EXPERIMENTS, 1):
        print_experiment_header(i, exp_info)
        
        exp_result = {
            'name': exp_info['name'],
            'description': exp_info['description'],
            'category': exp_info['category'],
            'status': 'unknown',
            'duration': 0,
            'error': None
        }
        
        # Check if outputs exist
        # 出力が存在するかチェック
        exp_name_lower = exp_info['name'].lower().replace('exp', 'exp')
        if exp_name_lower == 'beta':
            exp_name_lower = 'exp_beta'
        output_dir = Path(f"outputs/{exp_name_lower}")
        
        if skip_completed and output_dir.exists() and (output_dir / "results.csv").exists():
            print(f"  ⏭️  Skipping {exp_info['name']} (outputs exist)")
            print(f"  ⏭️  {exp_info['name']}をスキップ（出力が存在）")
            exp_result['status'] = 'skipped'
            results['skipped_count'] += 1
            results['experiments'].append(exp_result)
            continue
        
        # Run experiment
        # 実験を実行
        start_time = time.time()
        try:
            exp_info['function']()
            duration = time.time() - start_time
            
            exp_result['status'] = 'success'
            exp_result['duration'] = duration
            results['success_count'] += 1
            
            print()
            print(f"  ✅ {exp_info['name']} completed in {duration:.1f}s")
            print(f"  ✅ {exp_info['name']}が{duration:.1f}秒で完了")
            
        except Exception as e:
            duration = time.time() - start_time
            
            exp_result['status'] = 'failed'
            exp_result['duration'] = duration
            exp_result['error'] = str(e)
            results['failure_count'] += 1
            
            print()
            print(f"  ❌ {exp_info['name']} failed: {str(e)}")
            print(f"  ❌ {exp_info['name']}が失敗: {str(e)}")
        
        results['experiments'].append(exp_result)
    
    results['total_duration'] = time.time() - overall_start
    results['end_time'] = datetime.now().isoformat()
    
    return results


def print_summary(results: dict):
    """Print execution summary.
    
    実行要約を表示する。
    
    Args:
        results: Execution results dictionary
                実行結果辞書
    """
    print()
    print("=" * 80)
    print("EXECUTION SUMMARY / 実行要約")
    print("=" * 80)
    print()
    print(f"Total experiments / 合計実験数: {len(results['experiments'])}")
    print(f"✅ Successful / 成功: {results['success_count']}")
    print(f"❌ Failed / 失敗: {results['failure_count']}")
    print(f"⏭️  Skipped / スキップ: {results['skipped_count']}")
    print()
    print(f"Total duration / 合計時間: {results['total_duration']:.1f}s ({results['total_duration']/60:.1f}min)")
    print()
    
    # Show failed experiments
    # 失敗した実験を表示
    failed = [exp for exp in results['experiments'] if exp['status'] == 'failed']
    if failed:
        print("Failed experiments / 失敗した実験:")
        for exp in failed:
            print(f"  - {exp['name']}: {exp['error']}")
        print()
    
    # Show experiment durations
    # 実験時間を表示
    print("Experiment durations / 実験時間:")
    for exp in results['experiments']:
        if exp['status'] != 'skipped':
            status_icon = '✅' if exp['status'] == 'success' else '❌'
            print(f"  {status_icon} {exp['name']}: {exp['duration']:.1f}s")
    
    print()
    print("=" * 80)


def save_summary(results: dict):
    """Save execution summary to file.
    
    実行要約をファイルに保存する。
    
    Args:
        results: Execution results dictionary
                実行結果辞書
    """
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    summary_path = output_dir / "run_all_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Summary saved to / 要約を保存: {summary_path}")
    print()


def main():
    """Main execution function.
    
    メイン実行関数。
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run all E8 experiments / 全E8実験を実行'
    )
    parser.add_argument(
        '--skip-completed',
        action='store_true',
        help='Skip experiments with existing outputs / 既存出力がある実験をスキップ'
    )
    
    args = parser.parse_args()
    
    # Print header
    # ヘッダーを表示
    print_header()
    
    # Run all experiments
    # 全実験を実行
    results = run_all_experiments(skip_completed=args.skip_completed)
    
    # Print summary
    # 要約を表示
    print_summary(results)
    
    # Save summary
    # 要約を保存
    save_summary(results)
    
    # Exit with appropriate code
    # 適切な終了コードで終了
    if results['failure_count'] > 0:
        print("⚠️  Some experiments failed. Check errors above.")
        print("⚠️  いくつかの実験が失敗しました。上記のエラーを確認してください。")
        sys.exit(1)
    else:
        print("🎉 All experiments completed successfully!")
        print("🎉 全実験が正常に完了しました！")
        sys.exit(0)


if __name__ == "__main__":
    main()
