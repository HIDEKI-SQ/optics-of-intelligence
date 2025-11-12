#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

# ---- 固定設定（必要なら値だけ変えてください） ----
N = 1000         # サンプル数
D = 50           # 次元
SEED = 42        # 乱数シード
METRICS = ["euclidean", "cosine", "correlation"]
DTYPE = np.float64  # ここを np.float32 にすると差分がやや増えます（任意）

rng = np.random.default_rng(SEED)
A = rng.standard_normal((N, D)).astype(DTYPE)

# ---- 独立経路：NumPyのみで距離を実装 ----
def pairwise_dist_numpy(X: np.ndarray, metric: str) -> np.ndarray:
    """
    X: (N, D)
    return: square 距離行列 (N, N)  ＊pdistと独立に実装
    """
    if metric == "euclidean":
        # ||x-y|| = sqrt(||x||^2 + ||y||^2 - 2 x·y)
        sq_norm = np.einsum("ij,ij->i", X, X)
        D2 = sq_norm[:, None] + sq_norm[None, :] - 2.0 * (X @ X.T)
        np.maximum(D2, 0.0, out=D2)         # 数値誤差で負になるのを防ぐ
        D = np.sqrt(D2, out=D2)
        return D

    elif metric == "cosine":
        # 1 - cos = 1 - (x·y)/(||x|| ||y||)
        norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Z = X / norm
        S = Z @ Z.T                          # cos 類似度
        D = 1.0 - S
        return D

    elif metric == "correlation":
        # ピアソン相関に対応：各行の平均を引いて標準化 → cos と同型
        X0 = X - X.mean(axis=1, keepdims=True)
        std = X0.std(axis=1, ddof=0, keepdims=True) + 1e-12
        Z = X0 / std
        # 相関係数 = (1/D) * Z Z^T なので、距離は 1 - 相関
        S = (Z @ Z.T) / Z.shape[1]
        D = 1.0 - S
        return D

    else:
        raise ValueError(f"unknown metric: {metric}")

def condensed_from_square(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(M, k=1)
    return M[iu]

# ---- 2経路の差分 |Δ| を算出 ----
records = []
for metric in METRICS:
    # 経路①：SciPy pdist（condensed 形式）
    v1 = pdist(A, metric=metric)  # shape = (N*(N-1)/2,)

    # 経路②：NumPy実装で square → condensed
    Dsq = pairwise_dist_numpy(A, metric=metric)  # (N, N)
    v2 = condensed_from_square(Dsq)              # shape == v1

    diff = np.abs(v1 - v2).astype(np.float64)
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
