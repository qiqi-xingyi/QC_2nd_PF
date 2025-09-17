#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_fuse_rmsd_all.py
---------------------
Use ALL top-k candidates (here k=5) per PDB fragment to correlate fused energy Score
with RMSD. Produce:
  - summary_all_candidates.csv  (one row per candidate: 75*5 ~= 375 rows)
  - summary_best_per_pdb.csv    (best-by-Score per PDB: 75 rows)
  - correlation plots (rigid & scale)

Directory assumptions:
  project_root/
    results/
      hybrid_out/<pdb_id>/rerank.csv     # columns: cid,E_q,D_ss,D_phi_psi,Score
      quantum_rmsd/quantum_top5.csv      # columns include: pdb_id,tag,rmsd_rigid_A,rmsd_scale_A

Run in IDE: set CONFIG.project_root and run.
Run via CLI:
  python make_fuse_rmsd_all.py --project_root /path/to/QC_2nd_Protein_folding --out_dir results/rmsd_figures
"""

from __future__ import annotations
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: scipy for stats
try:
    from scipy.stats import pearsonr, spearmanr
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


# =========================
# ========= CONFIG ========
# =========================
class CONFIG:
    # 设置你的工程根目录（包含 results/hybrid_out 和 results/quantum_rmsd）
    project_root: str = r"/Users/yuqizhang/Desktop/Code/QC_2nd_Protein_folding"  # ←← 改成你的路径
    # 输出目录（相对 project_root 或绝对路径都可以）
    out_dir: str = "results/rmsd_figures"
    # 文件名
    summary_all_name: str = "summary_all_candidates.csv"
    summary_best_name: str = "summary_best_per_pdb.csv"
    # 相对路径
    hybrid_out_rel: str = os.path.join("results", "hybrid_out")
    rmsd_csv_rel: str = os.path.join("results", "quantum_rmsd", "quantum_top5.csv")
    # 只取 top_k（默认 5）；如果需要改更大/更小可在此设置
    top_k: int = 5


# =========================
# ========= I/O ===========
# =========================
def read_all_from_rerank(hybrid_out_dir: str, top_k: int) -> pd.DataFrame:
    """
    Load ALL candidates from results/hybrid_out/<pdb_id>/rerank.csv.
    Keep only rows whose cid looks like 'top_1'...'top_k' (if present).
    Return columns: pdb_id, cid, E_q, D_ss, D_phi_psi, Score
    """
    rows = []
    subdirs = [p for p in glob.glob(os.path.join(hybrid_out_dir, "*")) if os.path.isdir(p)]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {hybrid_out_dir}")

    valid_tags = {f"top_{i}" for i in range(1, top_k + 1)}

    for path in sorted(subdirs):
        pdb_id = os.path.basename(path)
        fcsv = os.path.join(path, "rerank.csv")
        if not os.path.exists(fcsv):
            print(f"[WARN] Missing rerank.csv: {fcsv}")
            continue
        try:
            df = pd.read_csv(fcsv)
        except Exception as e:
            print(f"[WARN] Failed to read {fcsv}: {e}")
            continue

        needed = {"cid", "Score"}
        if not needed.issubset(df.columns):
            print(f"[WARN] {fcsv} missing columns {needed}, found {set(df.columns)}")
            continue

        # 仅保留 top_1..top_k（如果 rerank.csv 含更多条目，避免混入其它 tag）
        if "cid" in df.columns:
            df = df[df["cid"].astype(str).isin(valid_tags)].copy()

        if df.empty:
            print(f"[WARN] No cid in {valid_tags} for {pdb_id}")
            continue

        # 选取需要的列；缺失的能量分量用 NaN 占位
        cols = ["cid", "Score"]
        for extra in ["E_q", "D_ss", "D_phi_psi"]:
            if extra in df.columns:
                cols.append(extra)

        df2 = df[cols].copy()
        df2.insert(0, "pdb_id", pdb_id)
        # 统一列名（若缺某列则补）
        for extra in ["E_q", "D_ss", "D_phi_psi"]:
            if extra not in df2.columns:
                df2[extra] = np.nan

        rows.append(df2)

    if not rows:
        raise RuntimeError("No valid rerank rows parsed.")
    out = pd.concat(rows, axis=0, ignore_index=True)
    # 排序（便于检查）
    out = out.sort_values(["pdb_id", "cid"]).reset_index(drop=True)
    return out


def load_rmsd_table(rmsd_csv: str) -> pd.DataFrame:
    if not os.path.exists(rmsd_csv):
        raise FileNotFoundError(f"RMSD csv not found: {rmsd_csv}")
    df = pd.read_csv(rmsd_csv)
    required = {"pdb_id", "tag", "rmsd_rigid_A", "rmsd_scale_A"}
    if not required.issubset(df.columns):
        raise ValueError(f"RMSD csv missing columns {required}, found {set(df.columns)}")
    return df


def merge_all_with_rmsd(all_df: pd.DataFrame, rmsd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge every candidate row (pdb_id, cid, Score, E_q, D_ss, D_phi_psi) with its RMSD by (pdb_id, tag=cid).
    """
    merged = all_df.merge(
        rmsd_df[["pdb_id", "tag", "rmsd_rigid_A", "rmsd_scale_A"]],
        left_on=["pdb_id", "cid"],
        right_on=["pdb_id", "tag"],
        how="left"
    ).drop(columns=["tag"])

    missing = merged[merged["rmsd_rigid_A"].isna()]
    if not missing.empty:
        print(f"[WARN] {missing.shape[0]} candidate rows missing RMSD match. "
              f"Examples: {missing[['pdb_id','cid']].head(5).to_dict(orient='records')}")
    return merged


def best_per_pdb_from_all(merged_all: pd.DataFrame) -> pd.DataFrame:
    """
    From the full candidate table, pick the minimal Score per pdb_id.
    """
    idx = merged_all.groupby("pdb_id")["Score"].idxmin()
    best = merged_all.loc[idx].copy().reset_index(drop=True)
    best = best.sort_values("pdb_id").reset_index(drop=True)
    return best


# =========================
# ======= PLOTTING ========
# =========================
def _compute_corr(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    res = {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    if x.size >= 2:
        if _SCIPY_OK:
            pr, pp = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
            res.update({"pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp})
        else:
            res["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
    return res, mask


def _fmt_p(p):
    return "n/a" if (p is None or (isinstance(p, float) and np.isnan(p))) else f"{p:.2e}"


def plot_corr(df: pd.DataFrame, x_col: str, y_col: str, out_png: str, out_pdf: str, title_suffix: str = ""):
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    stats, mask = _compute_corr(x, y)
    xm, ym = x[mask], y[mask]

    # Linear fit
    fit_x = fit_y = None
    fit_txt = ""
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
    # 注意：不指定颜色与样式，以便通用
    plt.scatter(xm, ym, s=28, alpha=0.85)
    if fit_x is not None:
        plt.plot(fit_x, fit_y, linewidth=2.0)
    plt.xlabel(r"Fused Energy $E_{\mathrm{fuse}}$")
    plt.ylabel(y_col.replace("_", r"\_"))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"[OK] Saved: {out_png}, {out_pdf}")


# =========================
# ======== RUNNER =========
# =========================
def ensure_dir(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def run(project_root: str, out_dir: str,
        hybrid_out_rel: str, rmsd_csv_rel: str,
        summary_all_name: str, summary_best_name: str,
        top_k: int):
    project_root = os.path.abspath(project_root)
    hybrid_out_dir = os.path.join(project_root, hybrid_out_rel)
    rmsd_csv = os.path.join(project_root, rmsd_csv_rel)

    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(project_root, out_dir)
    out_dir = ensure_dir(out_dir)

    # 1) 读取所有候选 (top_k) 的重排分数
    all_df = read_all_from_rerank(hybrid_out_dir, top_k=top_k)
    print(f"[INFO] Loaded rerank rows: {all_df.shape[0]}")

    # 2) 读取 RMSD 并合并
    rmsd_df = load_rmsd_table(rmsd_csv)
    merged_all = merge_all_with_rmsd(all_df, rmsd_df)

    # 3) 保存 375 点的完整汇总
    summary_all_csv = os.path.join(out_dir, summary_all_name)
    merged_all.to_csv(summary_all_csv, index=False)
    print(f"[OK] Saved full summary: {summary_all_csv}")

    # 4) 生成每个 PDB 的最优（便于主文摘要表）
    best_df = best_per_pdb_from_all(merged_all)
    summary_best_csv = os.path.join(out_dir, summary_best_name)
    best_df.to_csv(summary_best_csv, index=False)
    print(f"[OK] Saved per-PDB best: {summary_best_csv}")

    # 5) 相关性图（对 ALL candidates 作图，才能看出打分函数区分能力）
    plot_corr(merged_all, "Score", "rmsd_rigid_A",
              os.path.join(out_dir, "efuse_vs_rmsd_rigid_all.png"),
              os.path.join(out_dir, "efuse_vs_rmsd_rigid_all.pdf"),
              title_suffix="(rigid, all candidates)")
    plot_corr(merged_all, "Score", "rmsd_scale_A",
              os.path.join(out_dir, "efuse_vs_rmsd_scale_all.png"),
              os.path.join(out_dir, "efuse_vs_rmsd_scale_all.pdf"),
              title_suffix="(scale, all candidates)")


def parse_args():
    ap = argparse.ArgumentParser(description="Correlate fused energy with RMSD using ALL top-k candidates.")
    ap.add_argument("--project_root", type=str, default=None, help="Path to project root.")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory for CSV and figures.")
    ap.add_argument("--hybrid_out_rel", type=str, default=None, help="Relative path to results/hybrid_out.")
    ap.add_argument("--rmsd_csv_rel", type=str, default=None, help="Relative path to results/quantum_rmsd/quantum_top5.csv.")
    ap.add_argument("--summary_all_name", type=str, default=None, help="Filename for full summary CSV.")
    ap.add_argument("--summary_best_name", type=str, default=None, help="Filename for per-PDB best CSV.")
    ap.add_argument("--top_k", type=int, default=None, help="Use top-k candidates (default: 5).")
    return ap.parse_args()


if __name__ == "__main__":
    cfg = CONFIG()
    args = parse_args()

    project_root = args.project_root or cfg.project_root
    out_dir = args.out_dir or cfg.out_dir
    hybrid_out_rel = args.hybrid_out_rel or cfg.hybrid_out_rel
    rmsd_csv_rel = args.rmsd_csv_rel or cfg.rmsd_csv_rel
    summary_all_name = args.summary_all_name or cfg.summary_all_name
    summary_best_name = args.summary_best_name or cfg.summary_best_name
    top_k = args.top_k or cfg.top_k

    if not os.path.exists(project_root):
        raise FileNotFoundError(
            f"Project root not found: {project_root}\n"
            f"Please set CONFIG.project_root or pass --project_root."
        )

    run(project_root, out_dir, hybrid_out_rel, rmsd_csv_rel,
        summary_all_name, summary_best_name, top_k)
