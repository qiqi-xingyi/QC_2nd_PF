#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make a polished ALL-only scatter + linear regression plot for Score vs rmsd
with vivid colors and refined styling.

Input:
  results/rmsd_figures/summary_all_candidates.csv

Outputs:
  results/rmsd_figures/corr_plots/ALL_score_rmsd.png
  results/rmsd_figures/corr_plots/ALL_score_rmsd.pdf
  results/rmsd_figures/corr_plots/ALL_score_rmsd.svg
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


# ----------------------------- Styling helpers ----------------------------- #
def set_global_style():
    """Configure Matplotlib rcParams for a modern, high-contrast aesthetic."""
    mpl.rcParams.update({
        "figure.figsize": (8.8, 6.2),
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#111111",
        "axes.linewidth": 1.0,
        "axes.labelsize": 13,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelweight": "semibold",
        "axes.grid": True,
        "grid.color": "#9aa0a6",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "axes.prop_cycle": mpl.cycler(color=[
            "#2563EB",  # vivid blue
            "#EF4444",  # vivid red
            "#10B981",  # emerald
            "#F59E0B",  # amber
            "#7C3AED",  # violet
        ]),
    })


def beautify_axes(ax):
    """Subtle axis polishing."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


# ----------------------------- Core computation ---------------------------- #
def linear_fit_with_ci(x: np.ndarray, y: np.ndarray, x_line: np.ndarray, alpha: float = 0.05):
    """
    Fit y = a*x + b and compute the 95% CI for the mean prediction.
    Returns (a, b, y_line, y_low, y_high, r2, pearson_r).
    """
    n = len(x)
    if n < 2:
        raise ValueError("Not enough points for regression.")

    # OLS fit
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b

    # R^2 and Pearson r
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    pearson_r = np.corrcoef(x, y)[0, 1] if (np.std(x) > 0 and np.std(y) > 0) else np.nan

    # Mean-prediction CI (no SciPy; normal approx if n>=20)
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)
    dof = max(n - 2, 1)
    s = np.sqrt(ss_res / dof)
    t_crit = 1.96 if n >= 20 else 2.262  # ~95% critical value

    y_line = a * x_line + b
    if sxx > 0:
        se_mean = s * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / sxx)
        y_low = y_line - t_crit * se_mean
        y_high = y_line + t_crit * se_mean
    else:
        y_low = np.full_like(y_line, np.nan)
        y_high = np.full_like(y_line, np.nan)

    return a, b, y_line, y_low, y_high, r2, pearson_r


# ---------------------------------- Plotting -------------------------------- #
def plot_all_score_rmsd(x: np.ndarray, y: np.ndarray, outdir: Path):
    """
    Render a single ALL-only plot with vivid colors and premium layout.
    Saves PNG, PDF, and SVG.
    """
    # Colors
    c_scatter = "#2563EB"  # vivid blue
    c_line = "#EF4444"     # vivid red
    c_band = "#FCA5A5"     # soft red for CI (lighter)

    # Fit and CI
    x_min, x_max = float(np.min(x)), float(np.max(x))
    pad = 0.03 * (x_max - x_min if x_max > x_min else 1.0)
    x_line = np.linspace(x_min - pad, x_max + pad, 300)
    a, b, y_line, y_low, y_high, r2, r = linear_fit_with_ci(x, y, x_line)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Scatter: slightly transparent fill with edge strokes for clarity
    ax.scatter(x, y, s=46, alpha=0.85, c=c_scatter, edgecolor="#0B2E87", linewidth=0.6, label="Samples")

    # CI band: soft color with a bit of transparency
    ax.fill_between(x_line, y_low, y_high, alpha=0.22, color=c_band, label="95% CI")

    # Regression line: high-contrast color and thicker stroke
    ax.plot(x_line, y_line, color=c_line, linewidth=2.8, label="Linear fit")

    # Title / labels
    ax.set_title("ALL: Score vs rmsd", pad=10)
    ax.set_xlabel("Score")
    ax.set_ylabel("rmsd (Å)")

    # Legend in a clean location
    lg = ax.legend(loc="upper left", frameon=False, fontsize=11, handlelength=2.2)
    for t in lg.get_texts():
        t.set_weight("semibold")

    # Stats annotation box (top-right)
    text = (
        f"$R^2$ = {r2:.3f}\n"
        f"Pearson r = {r:.3f}\n"
        f"slope = {a:.4f}\n"
        f"intercept = {b:.4f}"
    )
    ax.text(
        0.98, 0.02, text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#111111", lw=1, alpha=0.95),
        fontsize=10,
    )

    beautify_axes(ax)
    fig.tight_layout()

    # Save multiple formats
    out_png = outdir / "ALL_score_rmsd.png"
    out_pdf = outdir / "ALL_score_rmsd.pdf"
    out_svg = outdir / "ALL_score_rmsd.svg"
    fig.savefig(out_png, dpi=340)
    fig.savefig(out_pdf)  # vector
    fig.savefig(out_svg)  # vector
    plt.close(fig)

    print(f"[OK] Wrote ALL plot:")
    print(f"  - {out_png}")
    print(f"  - {out_pdf}")
    print(f"  - {out_svg}")


# ----------------------------------- Main ----------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Polished ALL-only Score vs rmsd correlation plot.")
    parser.add_argument(
        "--input",
        type=str,
        default="results/rmsd_figures/summary_all_candidates.csv",
        help="Path to the aligned CSV."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/rmsd_figures/corr_plots",
        help="Directory to save the ALL plot."
    )
    args = parser.parse_args()

    set_global_style()

    input_csv = Path(args.input).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    for col in ("Score", "rmsd"):
        if col not in df.columns:
            raise ValueError("Input must contain 'Score' and 'rmsd' columns.")

    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["rmsd"]  = pd.to_numeric(df["rmsd"], errors="coerce")
    df = df.dropna(subset=["Score", "rmsd"]).copy()

    x = df["Score"].to_numpy(dtype=float)
    y = df["rmsd"].to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError("Need at least two points to plot ALL correlation.")

    plot_all_score_rmsd(x, y, outdir)


if __name__ == "__main__":
    main()
