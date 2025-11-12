import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import os

N, D, SEED = 1000, 50, 42
metrics = ["euclidean", "cosine", "correlation"]
rng = np.random.default_rng(SEED)
A = rng.standard_normal((N, D))

records = []
for metric in metrics:
    v1 = pdist(A, metric=metric)
    v2 = squareform(pdist(A, metric=metric)).ravel()
    diff = np.abs(v1 - v2)
    records.append({
        "metric": metric,
        "mean_diff": diff.mean(),
        "max_diff": diff.max(),
        "p95_diff": np.percentile(diff, 95),
        "n_pairs": len(diff)
    })

os.makedirs("outputs/m1", exist_ok=True)
pd.DataFrame(records).to_csv("outputs/m1/ssc_diff_summary.csv", index=False)
