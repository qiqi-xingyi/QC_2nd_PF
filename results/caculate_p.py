# --*-- conding:utf-8 --*--
# @time:10/6/25 18:04
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:caculate_p.py

#!/usr/bin/env python3
"""
Compute paired one-tailed t-tests comparing RMSD of baselines vs Hybrid.

Input CSV (path fixed to ./summary_rmsd.csv) must have columns:
  pdbid, af3, colab, quantum, hybrid

The script prints, for each comparison (AF3, ColabFold, Quantum-only vs Hybrid):
  - n (paired count)
  - mean difference (baseline - hybrid) with 95% CI
  - sd of differences and Cohen's dz
  - paired t statistic with one-tailed p value (H1: baseline > hybrid)
  - Wilcoxon signed-rank test (one-tailed, greater) as a robustness check
"""

import math
import pandas as pd
import numpy as np
from scipy import stats

CSV_PATH = "./summary_rmsd.csv"

def paired_t_one_tailed(x, y, greater=True):
    """
    Paired t-test with one-tailed p-value.
    If greater=True, tests H1: mean(x - y) > 0 (i.e., baseline > hybrid).
    Returns a dict with t, p_one_tailed, mean_diff, sd_diff, n, ci bounds, dz.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = x.size
    if n < 2:
        raise ValueError("Not enough paired observations after filtering NaNs.")
    d = x - y
    mean_diff = float(np.mean(d))
    sd_diff = float(np.std(d, ddof=1))
    # two sided t and p from scipy
    t_stat, p_two_sided = stats.ttest_rel(x, y, nan_policy='omit')
    # convert to one tailed according to direction
    if greater:
        # H1: mean(x - y) > 0
        p_one = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
    else:
        # H1: mean(x - y) < 0
        p_one = p_two_sided / 2 if t_stat < 0 else 1 - (p_two_sided / 2)
    # 95% CI for mean difference (two sided)
    df = n - 1
    se = sd_diff / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se
    # Cohen's dz for paired designs
    cohen_dz = mean_diff / sd_diff if sd_diff > 0 else float('inf')
    return {
        "n": n, "t": float(t_stat), "p_one_tailed": float(p_one),
        "mean_diff": mean_diff, "sd_diff": sd_diff,
        "ci95_low": float(ci_low), "ci95_high": float(ci_high),
        "cohen_dz": float(cohen_dz)
    }

def run_tests(df):
    # normalize column names
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"pdbid", "af3", "colab", "quantum", "hybrid"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {sorted(required)}; got {list(df.columns)}")

    # ensure numeric
    for col in ["af3", "colab", "quantum", "hybrid"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded {len(df)} rows from {CSV_PATH}.")
    print("After filtering finite pairs, sample sizes may differ per comparison.\n")

    comparisons = [
        ("AF3 vs Hybrid", "af3", "hybrid", True),
        ("ColabFold vs Hybrid", "colab", "hybrid", True),
        ("Quantum-only vs Hybrid", "quantum", "hybrid", True),
    ]

    for title, base_col, hyb_col, greater in comparisons:
        x = df[base_col].values
        y = df[hyb_col].values
        # paired t test
        res = paired_t_one_tailed(x, y, greater=greater)
        print(f"=== {title} ===")
        print(f"n = {res['n']}")
        print(f"mean difference (baseline - hybrid) = {res['mean_diff']:.3f} Å "
              f"[95% CI {res['ci95_low']:.3f}, {res['ci95_high']:.3f}]")
        print(f"sd of differences = {res['sd_diff']:.3f} Å, Cohen's dz = {res['cohen_dz']:.3f}")
        print(f"t({res['n']-1}) = {res['t']:.3f}, one-tailed p = {res['p_one_tailed']:.3e}")

        # Wilcoxon signed rank (one tailed, greater) as robustness check
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 2:
            stat, p_w = stats.wilcoxon(x[mask], y[mask], alternative='greater',
                                       zero_method='wilcox', correction=False, mode='approx')
            print(f"Wilcoxon signed-rank: W = {stat:.0f}, one-tailed p = {p_w:.3e} (n = {mask.sum()})")
        else:
            print("Wilcoxon signed-rank: insufficient pairs")
        print("")

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    run_tests(df)
