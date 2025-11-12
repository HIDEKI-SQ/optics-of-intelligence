# -*- coding: utf-8 -*-

import os
import pandas as pd

def test_cross_impl_threshold():
    """
    交差実装の差分が規格内（mean<0.07, max<0.10）に収まることを確認。
    ※ 本検証は数値的一致性の"下限"テストであり、
       実際の値は 1e-12 などの極小になるのが期待挙動。
    """
    csv_path = "outputs/m1/ssc_diff_summary.csv"
    assert os.path.exists(csv_path), f"missing CSV: {csv_path}"
    df = pd.read_csv(csv_path)

    assert (df["mean_diff"] < 0.07).all(), f"mean_diff out of spec:\n{df}"
    assert (df["max_diff"]  < 0.10).all(), f"max_diff out of spec:\n{df}"
