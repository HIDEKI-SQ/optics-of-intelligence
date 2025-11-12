#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ヘッドレス実行
import matplotlib.pyplot as plt

IN_CSV = "outputs/m1/ssc_diff_summary.csv"
OUT_DIR = "figs/m1"
OUT_PNG = os.path.join(OUT_DIR, "ssc_diff_bars.png")

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(IN_CSV)

ax = df.plot(
    x="metric",
    y=["mean_diff", "p95_diff", "max_diff"],
    kind="bar",
    legend=True,
    figsize=(6.4, 3.6)
)
ax.set_ylabel("Absolute difference")
ax.set_xlabel("Metric")
ax.set_title("Cross-implementation differences (condensed vs. square→condensed)")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print(f"✅ Saved {OUT_PNG}")
