#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ヘッドレス
import matplotlib.pyplot as plt

IN_CSV = "outputs/m1/ssc_diff_summary.csv"
OUT_DIR = "figs/m1"
OUT_PNG = os.path.join(OUT_DIR, "ssc_diff_bars.png")

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(IN_CSV)

# 極小値でもバーが見えるように微小εを足す（ゼロのみの時の視認性対策）
eps = 0.0
vals = df[["mean_diff", "p95_diff", "max_diff"]].to_numpy()
if np.all(vals == 0.0):
    eps = 1e-16
    df[["mean_diff", "p95_diff", "max_diff"]] = df[["mean_diff", "p95_diff", "max_diff"]] + eps

ax = df.plot(
    x="metric",
    y=["mean_diff", "p95_diff", "max_diff"],
    kind="bar",
    legend=True,
    figsize=(7.0, 3.8)
)
ax.set_ylabel("Absolute difference")
ax.set_xlabel("Metric")
title = "Cross-implementation differences (pdist vs. NumPy implementation)"
if eps > 0:
    title += "  [visualized with ε=1e-16]"
ax.set_title(title)

# スケールを自動調整（極小域を見やすく）
ymax = float(df[["mean_diff", "p95_diff", "max_diff"]].to_numpy().max())
if ymax <= 1e-12:
    ax.set_ylim(0, max(1e-12, ymax * 10.0))

# 値をバー上に表示（桁が小さいので科学記法）
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{height:.1e}", (p.get_x() + p.get_width()/2, height),
                ha="center", va="bottom", fontsize=8, rotation=90, xytext=(0,1), textcoords="offset points")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=220)
print(f"✅ Saved {OUT_PNG}")
