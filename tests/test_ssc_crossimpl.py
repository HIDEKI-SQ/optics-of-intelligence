# -*- coding: utf-8 -*-

import pandas as pd
import os

def test_cross_impl_threshold():
    """
    交差実装の差分が規格内に収まっていることを確認する。
    仕様：
      mean_diff < 0.07
      max_diff  < 0.10
    実装は outputs/m1/ssc_diff_summary.csv を読むだけ（CIで生成済み）。
    """
    csv_path = "outputs/m1/ssc_diff_summary.csv"
    assert os.path.exists(csv_path), f"missing CSV: {csv_path}"
    df = pd.read_csv(csv_path)

    assert (df["mean_diff"] < 0.07).all(), f"mean_diff out of spec:\n{df}"
    assert (df["max_diff"]  < 0.10).all(), f"max_diff out of spec:\n{df}"
