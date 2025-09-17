# --*-- conding:utf-8 --*--
# @time:9/17/25 01:37
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:make_fuse_rmsd.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_fuse_rmsd.py
-----------------
Aggregate the top-1 (lowest fused Score) per fragment from results/hybrid_out/<pdb>/rerank.csv,
merge with the corresponding RMSD from results/quantum_rmsd/quantum_top5.csv,
save a summary CSV, and plot correlation scatter plots between E_fuse and RMSD.

Run in IDE (no args needed): set CONFIG below and click Run.
Run via CLI (optional):
    python make_fuse_rmsd.py --project_root /path/to/QC_2nd_Protein_folding --out_dir results/rmsd_figures
"""

from __future__ import annotations
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Optional: scipy for statistics (Pearson/Spearman). If missing, Pearson via np.corrcoef; Spearman skipped. ----
try:
    from scipy.stats import pearsonr, spearmanr
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


# =========================
# ========= CONFIG ========
# =========================
class CONFIG:
    # 项目根目录（包含 results/hybrid_out 和 results/quantum_rmsd）
    project_root: str = r"/path/to/QC_2nd_Protein_folding"  # ←← 在 IDE 里把这行改成你的项目路径
    # 输出目录（相对 project_root 或绝对路径均可）
    out_dir: str = "results/rmsd_figures"
    # 汇总 CSV 文件名
    summary_name: str = "summary_fuse_rmsd.csv"
    # rerank.csv 所在目录
    hybrid_out_rel: str = os.path.join("results", "hybrid_out")
    # RMSD 文件
    rmsd_csv_rel: str = os.path.join("results", "quantum_rmsd", "quantum_top5.csv")


# =========================
# ====== CORE LOGIC =======
# =========================
def read_best_from_rerank(hybrid_out_dir: str) -> pd.DataFrame:
    """
    Scan results/hybrid_out/<pdb_id>/rerank.csv and pick the minimal 'Score'.
    Return columns: pdb_id, best_cid, best_score.
    """
    records = []
    subdirs = [p for p in glob.glob(os.path.join(hybrid_out_dir, "*")) if os.path.isdir(p)]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {hybrid_out_dir}")

    for path in sorted(subdirs):
        pdb_id = os.path.basename(path)
        rerank_csv = os.path.join(path, "rerank.csv")
        if not os.path.exists(rerank_csv):
            print(f"[WARN] Missing rerank.csv: {rerank_csv}")
            continue

        try:
            df = pd.read_csv(rerank_csv)
        except Exception as e:
            print(f"[WARN] Failed to read {rerank_csv}: {e}")
            continue

        # Expect at least these columns
        needed = {"cid", "Score"}
        if not needed.issubset(df.columns):
            print(f"[WARN] {rerank_csv} missing columns {needed}, found {set(df.columns)}")
            continue

        # Pick the minimal Score (top-1 after fusion)
        try:
            idx = df["Score"].astype(float).idxmin()
            row = df.loc[idx]
            best_cid = str(row["cid"])
            best_score = float(row["Score"])
        except Exception as e:
            print(f"[WARN] Could not find minimal Score in {rerank_csv}: {e}")
            continue

        records.append({"pdb_id": pdb_id, "best_cid": best_cid, "best_score": best_score})

    out = pd.DataFrame(records).sort_values("pdb_id").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"No valid rerank entries parsed from {hybrid_out_dir}")
    return out


def merge_with_rmsd(best_df: pd.DataFrame, quantum_rmsd_csv: str) -> pd.DataFrame:
    """
    Merge (pdb_id, best_cid, best_score) with RMSD from quantum_top5.csv by (pdb_id, tag).
    """
    if not os.path.exists(quantum_rmsd_csv):
        raise FileNotFoundError(f"RMSD csv not found: {quantum_rmsd_csv}")

    rmsd_df = pd.read_csv(quantum_rmsd_csv)
    required = {"pdb_id", "tag", "rmsd_rigid_A", "rmsd_scale_A"}
    if not required.issubset(rmsd_df.columns):
        raise ValueError(f"RMSD csv missing columns {required}, found {set(rmsd_df.columns)}")

    merged = best_df.merge(
        rmsd_df[list(required)],
        left_on=["pdb_id", "best_cid"], right_on=["pdb_id", "tag"],
        how="left"
    ).drop(columns=["tag"])

    # 提示哪些没匹配到（例如 rerank 里有 top_2，但 quantum_top5.csv 没有对应行）
    missing = merged[merged["rmsd_rigid_A"].isna()]
    if not missing.empty:
        miss_list = missing["pdb_id"].tolist()
        print(f"[WARN] {len(miss_list)} pdbs missing RMSD match in quantum_top5.csv: {miss_list[:10]}{' ...' if len(miss_list)>10 else ''}")

    return merged


def _compute_corr(x: np.ndarray, y: np.ndarray):
    """Compute Pearson & Spearman (if scipy present). Return dict and a finite mask."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    res = {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}

    if x.size >= 2:
        if _SCIPY_OK:
            pr, pp = pearsonr(x, y)
            res["pearson_r"], res["pearson_p"] = pr, pp
            sr, sp = spearmanr(x, y)
            res["spearman_rho"], res["spearman_p"] = sr, sp
        else:
            # Pearson only (no p-value)
            res["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
    return res, mask


def _fmt_p(pval: float) -> str:
    if pval is None or np.isnan(pval):
        return "n/a"
    return f"{pval:.2e}"


def plot_corr(df: pd.DataFrame, x_col: str, y_col: str, out_png: str, out_pdf: str, title_suffix: str = ""):
    """Scatter + linear fit; title shows Pearson/Spearman."""
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    stats, mask = _compute_corr(x, y)
    xm, ym = x[mask], y[mask]

    # Linear fit (if enough points)
    fit_x, fit_y, fit_txt = None, None, ""
    if xm.size >= 2:
        coef = np.polyfit(xm, ym, 1)
        fit_x = np.linspace(xm.min(), xm.max(), 200)
        fit_y = coef[0] * fit_x + coef[1]
        fit_txt = f"  (fit: y={coef[0]:.2f}x+{coef[1]:.2f})"

    title = (
        f"Correlation between $E_{{fuse}}$ and {y_col.replace('_', r'_')} {title_suffix}\n"
        f"Pearson r={stats['pearson_r']:.2f} (p={_fmt_p(stats['pearson_p'])}), "
        f"Spearman $\\rho$={stats['spearman_rho']:.2f} (p={_fmt_p(stats['spearman_p'])}){fit_txt}"
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(xm, ym, s=30, alpha=0.85)  # 使用默认配色
    if fit_x is not None:
        plt.plot(fit_x, fit_y, linewidth=2.0)
    plt.xlabel(r"Fused Energy $E_{\mathrm{fuse}}$")
    plt.ylabel(y_col.replace("_", r"\_"))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"[OK] Saved plots: {out_png}, {out_pdf}")


def ensure_dir(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


# =========================
# ======== RUNNER =========
# =========================
def run(project_root: str, out_dir: str, summary_name: str,
        hybrid_out_rel: str, rmsd_csv_rel: str):
    project_root = os.path.abspath(project_root)
    hybrid_out_dir = os.path.join(project_root, hybrid_out_rel)
    rmsd_csv = os.path.join(project_root, rmsd_csv_rel)

    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(project_root, out_dir)
    out_dir = ensure_dir(out_dir)

    # 1) 读取每个 pdb 的重排最优（Score 最小）
    best_df = read_best_from_rerank(hybrid_out_dir)
    print(f"[INFO] Parsed best rerank from {len(best_df)} fragments.")

    # 2) 合并 RMSD
    merged = merge_with_rmsd(best_df, rmsd_csv)

    # 3) 保存汇总
    summary_csv = os.path.join(out_dir, summary_name)
    merged.to_csv(summary_csv, index=False)
    print(f"[OK] Summary saved: {summary_csv}")
    print(merged.head(8))

    # 4) 画相关性图（rigid 与 scale）
    plot_corr(merged, "best_score", "rmsd_rigid_A",
              os.path.join(out_dir, "efuse_vs_rmsd_rigid.png"),
              os.path.join(out_dir, "efuse_vs_rmsd_rigid.pdf"),
              title_suffix="(rigid)")
    plot_corr(merged, "best_score", "rmsd_scale_A",
              os.path.join(out_dir, "efuse_vs_rmsd_scale.png"),
              os.path.join(out_dir, "efuse_vs_rmsd_scale.pdf"),
              title_suffix="(scale)")


def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate fused energy (top-1) and RMSD; plot correlation.")
    ap.add_argument("--project_root", type=str, default=None, help="Path to QC_2nd_Protein_folding root.")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory for CSV and figures.")
    ap.add_argument("--summary_name", type=str, default=None, help="Summary CSV name.")
    ap.add_argument("--hybrid_out_rel", type=str, default=None, help="Relative path to results/hybrid_out.")
    ap.add_argument("--rmsd_csv_rel", type=str, default=None, help="Relative path to results/quantum_rmsd/quantum_top5.csv.")
    return ap.parse_args()


if __name__ == "__main__":
    # 1) IDE 直接运行：使用 CONFIG
    cfg = CONFIG()

    # 2) 命令行参数（如果提供则覆盖 CONFIG）
    args = parse_args()
    project_root = args.project_root or cfg.project_root
    out_dir = args.out_dir or cfg.out_dir
    summary_name = args.summary_name or cfg.summary_name
    hybrid_out_rel = args.hybrid_out_rel or cfg.hybrid_out_rel
    rmsd_csv_rel = args.rmsd_csv_rel or cfg.rmsd_csv_rel

    if not os.path.exists(project_root):
        raise FileNotFoundError(
            f"Project root not found: {project_root}\n"
            f"Please set CONFIG.project_root or pass --project_root."
        )

    run(project_root, out_dir, summary_name, hybrid_out_rel, rmsd_csv_rel)
