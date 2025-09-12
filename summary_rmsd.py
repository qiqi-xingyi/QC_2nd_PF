# --*-- conding:utf-8 --*--
# @time:9/12/25 02:27
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:summary_rmsd.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RMSD visualization and statistics for four methods across multiple proteins.

Input:
  results/summary_rmsd.csv  with columns: pdbid,af3,colab,quantum,hybrid
Outputs (under results/rmsd_figures/):
  - fig1_violin_jitter.(pdf|svg)
  - fig2_ecdf.(pdf|svg)
  - fig3_improvement_hist.(pdf|svg)
  - fig4a_avg_rank.(pdf|svg)
  - fig4b_win_counts.(pdf|svg)
  - fig5_rank_heatmap.(pdf|svg)
  - metrics_aggregate.csv
  - metrics_coverage.csv
  - metrics_ranks_wins.csv
  - metrics_improvements_vs_hybrid.csv
  - README.txt
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import scipy for Wilcoxon; handle absence gracefully.
try:
    from scipy.stats import wilcoxon
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------- Config ----------
CSV_PATH = "results/summary_rmsd.csv"
OUT_DIR = "results/rmsd_figures"
METHODS = ["af3", "colab", "quantum", "hybrid"]
DISPLAY_NAMES = {"af3": "AF3", "colab": "ColabFold", "quantum": "Quantum", "hybrid": "Hybrid"}
THRESHOLDS = (3.0, 4.0, 5.0)  # Å thresholds for coverage
RANDOM_SEED = 42


def ensure_io() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        sys.exit(f"[ERROR] CSV not found: {CSV_PATH}")
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    needed = ["pdbid"] + METHODS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Missing required columns: {missing}")
    df = df[needed].dropna()
    if df.empty:
        sys.exit("[ERROR] No data after dropping NaNs.")
    return df


def savefig(base: str):
    for ext in ("pdf", "svg"):
        plt.savefig(os.path.join(OUT_DIR, f"{base}.{ext}"), bbox_inches="tight")


def ecdf_values(arr: np.ndarray):
    y = np.sort(arr)
    x_step = np.concatenate(([y[0]], y))
    y_step = np.arange(len(y) + 1) / len(y)
    return x_step, y_step


def coverage(arr: np.ndarray, thr: float) -> float:
    return float((arr <= thr).mean())


def figure_violin_jitter(df: pd.DataFrame):
    plt.figure(figsize=(6.4, 3.8))
    data = [df[m].to_numpy(dtype=float) for m in METHODS]

    plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    # Quartiles and median markers
    for i, m in enumerate(METHODS, start=1):
        y = df[m].to_numpy(dtype=float)
        q1, med, q3 = np.percentile(y, [25, 50, 75])
        plt.scatter([i], [med], marker="o", zorder=3)
        plt.vlines(i, q1, q3, linewidth=3)

    # Jittered points
    rng = np.random.default_rng(RANDOM_SEED)
    for i, y in enumerate(data, start=1):
        xj = i + (rng.random(len(y)) - 0.5) * 0.22
        plt.plot(xj, y, linestyle="none", marker=".", alpha=0.35)

    plt.xticks(range(1, len(METHODS) + 1), [DISPLAY_NAMES[m] for m in METHODS])
    plt.ylabel("RMSD (Å)")
    plt.title("RMSD distribution across proteins (lower is better)")
    plt.grid(alpha=0.25, axis="y")
    savefig("fig1_violin_jitter")
    plt.close()


def figure_ecdf(df: pd.DataFrame):
    plt.figure(figsize=(6.4, 3.8))
    for m in METHODS:
        x_step, y_step = ecdf_values(df[m].to_numpy(dtype=float))
        plt.step(x_step, y_step, where="post", label=DISPLAY_NAMES[m])
    for thr in THRESHOLDS:
        plt.axvline(thr, linestyle="--", alpha=0.2)
    plt.xlabel("RMSD (Å)")
    plt.ylabel("Proportion ≤ x")
    plt.title("ECDF of RMSD (higher curve is better)")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    savefig("fig2_ecdf")
    plt.close()


def figure_improvement_hist(df: pd.DataFrame) -> pd.DataFrame:
    plt.figure(figsize=(6.4, 3.8))
    bins = 20
    rows = []
    for m in ["af3", "colab", "quantum"]:
        delta = df[m].to_numpy(dtype=float) - df["hybrid"].to_numpy(dtype=float)
        plt.hist(delta, bins=bins, histtype="step", label=f"{DISPLAY_NAMES[m]} − Hybrid")
        rows.append({
            "method": DISPLAY_NAMES[m],
            "win_rate_vs_hybrid": float((delta > 0).mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta))
        })
    plt.axvline(0.0, linestyle="--", alpha=0.4)
    plt.xlabel("RMSD difference (method − Hybrid)  [Å]")
    plt.ylabel("Count")
    plt.title("Per-sample improvements of Hybrid (right = worse than Hybrid)")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    savefig("fig3_improvement_hist")
    plt.close()
    return pd.DataFrame(rows)


def figure_avg_rank_and_wins(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    vals = df[METHODS].to_numpy(dtype=float)
    ranks = np.argsort(np.argsort(vals, axis=1), axis=1) + 1  # ascending rank
    avg_rank = ranks.mean(axis=0)
    row_mins = vals.min(axis=1, keepdims=True)
    win_counts = (vals == row_mins).sum(axis=0)

    # Average rank bar
    plt.figure(figsize=(5.4, 3.2))
    plt.bar(range(len(METHODS)), avg_rank)
    plt.xticks(range(len(METHODS)), [DISPLAY_NAMES[m] for m in METHODS])
    plt.ylabel("Average rank (lower is better)")
    plt.title("Average rank across proteins")
    plt.grid(alpha=0.25, axis="y")
    savefig("fig4a_avg_rank")
    plt.close()

    # Win counts bar
    plt.figure(figsize=(5.4, 3.2))
    plt.bar(range(len(METHODS)), win_counts)
    plt.xticks(range(len(METHODS)), [DISPLAY_NAMES[m] for m in METHODS])
    plt.ylabel("#Proteins with best (lowest) RMSD")
    plt.title("Win count (ties counted for all winners)")
    plt.grid(alpha=0.25, axis="y")
    savefig("fig4b_win_counts")
    plt.close()

    rank_df = pd.DataFrame({
        "method": [DISPLAY_NAMES[m] for m in METHODS],
        "avg_rank": avg_rank,
        "win_count": win_counts
    })
    return ranks, rank_df


def figure_rank_heatmap(df: pd.DataFrame, ranks: np.ndarray):
    order = np.argsort(df["hybrid"].to_numpy(dtype=float))  # best Hybrid first
    rank_mat = ranks[order, :]

    plt.figure(figsize=(6.2, 6.2))
    plt.imshow(rank_mat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Rank (1=best)")
    # show ~15 y-ticks max
    n = len(order)
    step = max(1, n // 15)
    idxs = list(range(0, n, step))
    plt.yticks(idxs, df["pdbid"].to_numpy()[order][idxs])
    plt.xticks(range(len(METHODS)), [DISPLAY_NAMES[m] for m in METHODS])
    plt.title("Per-protein rank heatmap (sorted by Hybrid)")
    savefig("fig5_rank_heatmap")
    plt.close()


def compute_summaries_and_tests(df: pd.DataFrame) -> dict:
    out = {}

    # Aggregate stats
    agg_rows = []
    cov_rows = []
    for m in METHODS:
        arr = df[m].to_numpy(dtype=float)
        agg_rows.append({
            "method": DISPLAY_NAMES[m],
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(arr.size),
        })
        cov = {"method": DISPLAY_NAMES[m]}
        for thr in THRESHOLDS:
            cov[f"cov_le_{thr}A"] = coverage(arr, thr)
        cov_rows.append(cov)

    out["aggregate"] = pd.DataFrame(agg_rows)
    out["coverage"] = pd.DataFrame(cov_rows)

    # Wilcoxon tests (paired): H1: Hybrid < other
    tests = []
    if SCIPY_OK:
        base = "hybrid"
        for m in ["quantum", "af3", "colab"]:
            stat, p = wilcoxon(
                df[base].to_numpy(dtype=float),
                df[m].to_numpy(dtype=float),
                alternative="less"
            )
            tests.append({"comparison": f"Hybrid < {DISPLAY_NAMES[m]}", "p_value": float(p)})
    else:
        tests.append({"comparison": "scipy.stats.wilcoxon", "p_value": None})

    out["tests"] = pd.DataFrame(tests)
    return out


def write_tables(tables: dict, rank_df: pd.DataFrame, improve_df: pd.DataFrame):
    tables["aggregate"].to_csv(os.path.join(OUT_DIR, "metrics_aggregate.csv"), index=False)
    tables["coverage"].to_csv(os.path.join(OUT_DIR, "metrics_coverage.csv"), index=False)
    rank_df.to_csv(os.path.join(OUT_DIR, "metrics_ranks_wins.csv"), index=False)
    improve_df.to_csv(os.path.join(OUT_DIR, "metrics_improvements_vs_hybrid.csv"), index=False)
    if "tests" in tables:
        tables["tests"].to_csv(os.path.join(OUT_DIR, "metrics_tests.csv"), index=False)


def write_readme(df: pd.DataFrame, tables: dict, rank_df: pd.DataFrame, improve_df: pd.DataFrame):
    lines = []
    lines.append("RMSD Visualization & Statistics\n")
    lines.append(f"Input CSV: {CSV_PATH}\n")
    lines.append(f"Total proteins (rows): {len(df)}\n")

    lines.append("\nAggregate statistics (mean/median/std/min/max):\n")
    for _, r in tables["aggregate"].iterrows():
        lines.append(
            f"  {r['method']:>10s}  mean={r['mean']:.3f}  median={r['median']:.3f}  "
            f"std={r['std']:.3f}  min={r['min']:.3f}  max={r['max']:.3f}"
        )

    lines.append("\nCoverage (proportion ≤ threshold):\n")
    for _, r in tables["coverage"].iterrows():
        cov_str = "  ".join([f"≤{thr}Å={r[f'cov_le_{thr}A']:.3f}" for thr in THRESHOLDS])
        lines.append(f"  {r['method']:>10s}  {cov_str}")

    lines.append("\nAverage rank and wins:\n")
    for _, r in rank_df.iterrows():
        lines.append(
            f"  {r['method']:>10s}  avg_rank={r['avg_rank']:.3f}  win_count={int(r['win_count'])}"
        )

    lines.append("\nPaired Wilcoxon tests (H1: Hybrid < other):\n")
    if SCIPY_OK:
        tests_df = tables.get("tests", pd.DataFrame())
        for _, t in tests_df.iterrows():
            lines.append(f"  {t['comparison']}: p = {t['p_value']:.3e}")
    else:
        lines.append("  scipy not installed; skipped Wilcoxon tests.")

    lines.append("\nImprovements vs Hybrid (method − Hybrid):\n")
    for _, r in improve_df.iterrows():
        lines.append(
            f"  {r['method']:>10s}  win_rate_vs_hybrid={r['win_rate_vs_hybrid']:.3f}  "
            f"mean_delta={r['mean_delta']:.3f}  median_delta={r['median_delta']:.3f}"
        )

    with open(os.path.join(OUT_DIR, "README.txt"), "w") as f:
        f.write("\n".join(lines))


def main():
    df = ensure_io()

    # Figures
    figure_violin_jitter(df)
    figure_ecdf(df)
    improve_df = figure_improvement_hist(df)
    ranks, rank_df = figure_avg_rank_and_wins(df)
    figure_rank_heatmap(df, ranks)

    # Tables & tests
    tables = compute_summaries_and_tests(df)
    write_tables(tables, rank_df, improve_df)
    write_readme(df, tables, rank_df, improve_df)

    print(f"[OK] Figures and stats saved under: {OUT_DIR}")
    if not SCIPY_OK:
        print("[INFO] scipy not found. Install scipy to enable Wilcoxon tests: pip install scipy")


if __name__ == "__main__":
    main()
