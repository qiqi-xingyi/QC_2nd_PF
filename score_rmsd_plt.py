# --*-- conding:utf-8 --*--
# @time:9/17/25 02:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:score_rmsd_plt.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make a single, polished ALL-only scatter + linear regression plot for Score vs rmsd.

Input:
  results/rmsd_figures/summary_all_candidates_aligned.csv

Output:
  results/rmsd_figures/corr_plots/ALL_score_rmsd.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


def linear_fit_with_ci(x: np.ndarray, y: np.ndarray, x_line: np.ndarray, alpha: float = 0.05):
    """
    Fit y = a*x + b. Return (a, b, y_line, y_low, y_high, r2, pearson_r).
    95% CI for the regression mean is computed via classic OLS formulas.
    """
    n = len(x)
    if n < 2:
        raise ValueError("Not enough points for regression.")

    # Fit
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b

    # R^2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Pearson r
    pearson_r = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan

    # CI for the mean prediction
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)
    dof = max(n - 2, 1)
    s = np.sqrt(ss_res / dof)  # residual std

    # t critical value (approx via normal if n large)
    # For simplicity and to avoid scipy, use 1.96 as ~95% for n>=20.
    t_crit = 1.96 if n >= 20 else 2.262  # n<20 fallback ~ t_{0.975,18}

    y_line = a * x_line + b
    se_mean = s * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / sxx) if sxx > 0 else np.full_like(x_line, np.nan)
    y_low = y_line - t_crit * se_mean
    y_high = y_line + t_crit * se_mean

    return a, b, y_line, y_low, y_high, r2, pearson_r


def beautify_axes(ax):
    """Minimal aesthetic touches without specifying colors."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Plot ALL-only Score vs rmsd correlation.")
    parser.add_argument(
        "--input",
        type=str,
        default="results/rmsd_figures/summary_all_candidates_aligned.csv",
        help="Path to the aligned CSV."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/rmsd_figures/corr_plots",
        help="Directory to save the ALL plot."
    )
    args = parser.parse_args()

    input_csv = Path(args.input).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)

    df = pd.read_csv(input_csv)
    for col in ("Score", "rmsd"):
        if col not in df.columns:
            raise ValueError("Input must contain 'Score' and 'rmsd' columns.")

    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["rmsd"] = pd.to_numeric(df["rmsd"], errors="coerce")
    df = df.dropna(subset=["Score", "rmsd"])

    x = df["Score"].to_numpy(dtype=float)
    y = df["rmsd"].to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError("Need at least two points to plot ALL correlation.")

    # Build x-grid for a smooth line
    x_min, x_max = float(np.min(x)), float(np.max(x))
    pad = 0.02 * (x_max - x_min if x_max > x_min else 1.0)
    x_line = np.linspace(x_min - pad, x_max + pad, 200)

    a, b, y_line, y_low, y_high, r2, r = linear_fit_with_ci(x, y, x_line)

    # Figure
    fig = plt.figure(figsize=(7.5, 5.8), dpi=110)
    ax = fig.add_subplot(111)

    # Scatter (default styles), slightly larger markers for readability
    ax.scatter(x, y, s=36, alpha=0.85)

    # Regression band and line (default styles)
    ax.fill_between(x_line, y_low, y_high, alpha=0.18)
    ax.plot(x_line, y_line, linewidth=2.5)

    # Labels and title
    ax.set_title("ALL: Score vs rmsd", fontsize=14, pad=10)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("rmsd (Å)", fontsize=12)

    # Stats box
    text = (
        f"$R^2$ = {r2:.3f}\n"
        f"Pearson r = {r:.3f}\n"
        f"slope = {a:.4f}\n"
        f"intercept = {b:.4f}"
    )
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=1, alpha=0.9),
        fontsize=10,
    )

    beautify_axes(ax)
    fig.tight_layout()

    out_png = outdir / "ALL_score_rmsd.png"
    fig.savefig(out_png, dpi=320, transparent=False)
    plt.close(fig)

    print(f"[OK] Wrote overall plot: {out_png}")


if __name__ == "__main__":
    main()
