import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import os

N, D, SEED = 1000, 50, 42
metrics = ["euclidean", "cosine", "correlation"]

rng = np.random.default_rng(SEED)
A = rng.standard_normal((N, D))
# 任意：行正規化したい場合は次の2行を有効化
# A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)

def condensed_from_square(M: np.ndarray) -> np.ndarray:
    """square 距離行列から condensed ベクトルを取り出す（上三角, k=1）"""
    iu = np.triu_indices_from(M, k=1)
    return M[iu]

records = []
for metric in metrics:
    # 実装①：直接 condensed を得る
    v1 = pdist(A, metric=metric)  # shape = (N*(N-1)/2,)

    # 実装②：いったん square を作ってから condensed に戻す
    D = squareform(v1)            # shape = (N, N)
    v2 = condensed_from_square(D) # shape を v1 に合わせる

    diff = np.abs(v1 - v2)
    records.append({
        "metric": metric,
        "mean_diff": float(diff.mean()),
        "max_diff": float(diff.max()),
        "p95_diff": float(np.percentile(diff, 95)),
        "n_pairs": int(diff.size)
    })

os.makedirs("outputs/m1", exist_ok=True)
pd.DataFrame(records).to_csv("outputs/m1/ssc_diff_summary.csv", index=False)
print("✅ Finished: outputs/m1/ssc_diff_summary.csv generated")
