# --*-- conding:utf-8 --*--
# @time:9/7/25 23:54
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plot_all_rerank.py

"""
Batch plot stacked bar charts for all PDBs under E_rerank/out.

For each <pdbid>:
- Read E_rerank/out/<pdbid>/rerank.csv
- Plot stacked bars:
    segment 1 = E_q
    segment 2 = D_ss
    segment 3 = D_phi_psi
- Save to <out_fig_dir>/<pdbid>_rerank_stacked.png

Usage:
    python tools/plot_all_rerank.py \
        --in_root E_rerank/out \
        --out_fig_dir figs_rerank

Notes:
- If a column (e.g., D_ss) is missing, it will be treated as zeros and a warning is printed.
- Candidate order is kept as in rerank.csv (which is already sorted by Score in your pipeline).
"""

import os
import json
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _safe_read_metadata(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def plot_one_pdb(pdb_dir: Path, out_fig_dir: Path) -> Path:
    """
    Plot a stacked bar chart for a single pdb folder.
    Returns the path to the saved figure.
    """
    pdbid = pdb_dir.name
    rerank_path = pdb_dir / "rerank.csv"
    meta_path = pdb_dir / "metadata.json"

    df = _safe_read_csv(rerank_path)
    meta = _safe_read_metadata(meta_path)

    # Ensure required columns exist; fill missing with zeros (warn).
    required = ["cid", "E_q", "D_ss", "D_phi_psi"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] {pdbid}: columns missing in rerank.csv -> {missing}. Will fill with zeros.")
        for c in missing:
            if c == "cid":
                # If even 'cid' is missing, fabricate candidate ids
                df["cid"] = [f"cand_{i+1}" for i in range(len(df))]
            else:
                df[c] = 0.0

    # Coerce numeric & handle NaNs
    for c in ["E_q", "D_ss", "D_phi_psi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(0.0)

    # Height check: if Score exists, we don't need it for plotting; stacked bars show the decomposition.
    # But we can use metadata to enrich the title.
    title_bits = [f"{pdbid} — stacked score composition"]
    if isinstance(meta, dict):
        p = meta.get("params", {})
        if p:
            # show a compact param summary
            ss_mode = p.get("ss_mode", "ss3")
            dist = p.get("dist", "ce")
            alpha = p.get("alpha", 1.0)
            beta = p.get("beta", 1.0)
            gamma = p.get("gamma", 1.0)
            title_bits.append(f"(ss={ss_mode}, dist={dist}, α={alpha}, β={beta}, γ={gamma})")
    title = " ".join(title_bits)

    # Plot
    x = np.arange(len(df))
    bottom = np.zeros(len(df), dtype=float)

    parts = [
        ("E_q", "steelblue"),
        ("D_ss", "seagreen"),
        ("D_phi_psi", "darkorange"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in parts:
        ax.bar(x, df[label].to_numpy(float), bottom=bottom, color=color, edgecolor="black", label=label)
        bottom += df[label].to_numpy(float)

    # X ticks & labels
    ax.set_xticks(x)
    ax.set_xticklabels(df["cid"].astype(str).tolist(), rotation=0)
    ax.set_ylabel("Score contribution")
    ax.set_title(title)
    ax.legend()

    # Add a thin line annotation for total on top of each bar (optional but nice)
    totals = bottom
    for i, v in enumerate(totals):
        ax.text(i, v + max(0.02, 0.01 * totals.max()), f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()

    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_fig_dir / f"{pdbid}_rerank_stacked.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[OK] Saved figure -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Batch plot stacked bars from E_rerank/out/*/rerank.csv")
    parser.add_argument("--in_root", type=str, default="E_rerank/out",
                        help="Root folder containing per-pdb output subfolders")
    parser.add_argument("--out_fig_dir", type=str, default="figs_rerank",
                        help="Where to save all figures")
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="Optional list of pdbids to include (others are skipped)")
    args = parser.parse_args()

    in_root = Path(args.in_root).resolve()
    out_fig_dir = Path(args.out_fig_dir).resolve()
    to_include = set([s.strip() for s in args.only]) if args.only else None

    if not in_root.is_dir():
        raise NotADirectoryError(f"in_root does not exist or is not a directory: {in_root}")

    subdirs = sorted([p for p in in_root.iterdir() if p.is_dir()])
    if not subdirs:
        print(f"[WARN] No subfolders found under {in_root}")
        return

    ok, fail = 0, 0
    for pdb_dir in subdirs:
        pdbid = pdb_dir.name
        if to_include and pdbid not in to_include:
            continue
        try:
            plot_one_pdb(pdb_dir, out_fig_dir)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[ERR] {pdbid}: {e}")
            traceback.print_exc()

    print(f"[DONE] Plotted {ok} figure(s). Failed: {fail}. Output dir: {out_fig_dir}")


if __name__ == "__main__":
    main()
