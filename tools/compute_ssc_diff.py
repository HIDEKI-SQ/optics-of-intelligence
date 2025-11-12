#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# ---- 設定（必要なら固定値でOK） ----
N = 1000         # サンプル点（アイテム）数
D = 50           # 埋め込み次元
SEED = 42        # 乱数シード（決定論）
METRICS = ["euclidean", "cosine", "correlation"]

# ---- データ生成（決定論） ----
rng = np.random.default_rng(SEED)
A = rng.standard_normal((N, D))
# 行正規化が必要なら以下の1行を有効化してください（任意）
# A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)

def condensed_from_square(M: np.ndarray) -> np.ndarray:
    """square距離行列から condensed ベクトル（上三角, k=1）を取り出す"""
    iu = np.triu_indices_from(M, k=1)
    return M[iu]

# ---- 計算（2経路の差分 |Δ| ）----
records = []
for metric in METRICS:
    # 実装①：直接 condensed を得る
    v1 = pdist(A, metric=metric)                 # shape = (N*(N-1)/2,)

    # 実装②：square化→上三角抽出で condensed に揃える
    Dmat = squareform(v1)                         # shape = (N, N)
    v2 = condensed_from_square(Dmat)              # shape == v1

    # 差分（理論上一致：0.0）
    diff = np.abs(v1 - v2)
    records.append({
        "metric": metric,
        "mean_diff": float(diff.mean()),
        "max_diff": float(diff.max()),
        "p95_diff": float(np.percentile(diff, 95)),
        "n_pairs": int(diff.size)
    })

# ---- 出力 ----
os.makedirs("outputs/m1", exist_ok=True)
df = pd.DataFrame(records, columns=["metric", "mean_diff", "max_diff", "p95_diff", "n_pairs"])
df.to_csv("outputs/m1/ssc_diff_summary.csv", index=False)
print("✅ Finished: outputs/m1/ssc_diff_summary.csv generated")
