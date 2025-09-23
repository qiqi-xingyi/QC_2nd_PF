# --*-- conding:utf-8 --*--
# @time:9/21/25 02:39
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:get_tex_table.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pandas as pd

ORDER = ["top_1", "top_2", "top_3", "top_4", "top_5"]


def fmt_float2(x):
    try:
        v = float(x)
        return f"{v:.2f}"
    except Exception:
        return ""


def fmt_int(x):
    try:
        v = float(x)
        return f"{int(round(v))}"
    except Exception:
        return ""


def build_row(group: pd.DataFrame) -> str:
    group = group.copy()
    # Ensure ordered candidates and one row per cid
    group = group.set_index("cid").reindex(ORDER)
    # Extract shared metadata
    pdbid = str(group["pdbid"].iloc[0])
    try:
        length = int(group["length"].iloc[0])
    except Exception:
        length = ""
    try:
        qubits = int(group["qubits"].iloc[0])
    except Exception:
        qubits = ""
    try:
        depth = int(group["depth"].iloc[0])
    except Exception:
        depth = ""

    cells = [pdbid, str(length), str(qubits), str(depth)]
    for cid in ORDER:
        if cid in group.index:
            r = group.loc[cid]
            rmsd = fmt_float2(r.get("rmsd", ""))
            score = fmt_float2(r.get("Score", ""))
            energy = fmt_int(r.get("energy", ""))
        else:
            rmsd = score = energy = ""
        cells += [rmsd, score, energy]

    return " & ".join(cells) + r" \\"


def write_longtable(df: pd.DataFrame, out_tex: str):
    header = r"""
{\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{0.9}
\begin{longtable}{lrrrrrrrrrrrrrrrrrr}
\caption{Per-fragment summary with fused score (Score), RMSD, and quantum energy (integer) for the top-5 candidates. Left block reports chain length, qubit count, and circuit depth; each candidate block lists \textit{rmsd}, \textit{Score}, and \textit{Energy}.}
\label{tab:fused_rmsd_energy}\\
\toprule
\textbf{pdbid} & \textbf{Len} & \textbf{Qubits} & \textbf{Depth} &
\multicolumn{3}{c}{\textbf{Candidate1}} &
\multicolumn{3}{c}{\textbf{Candidate2}} &
\multicolumn{3}{c}{\textbf{Candidate3}} &
\multicolumn{3}{c}{\textbf{Candidate4}} &
\multicolumn{3}{c}{\textbf{Candidate5}} \\
\cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13} \cmidrule(lr){14-16} \cmidrule(lr){17-19}
 &  &  &  &
\textit{rmsd} & \textit{Score} & \textit{Energy} &
\textit{rmsd} & \textit{Score} & \textit{Energy} &
\textit{rmsd} & \textit{Score} & \textit{Energy} &
\textit{rmsd} & \textit{Score} & \textit{Energy} &
\textit{rmsd} & \textit{Score} & \textit{Energy} \\
\midrule
\endfirsthead

\toprule
\textbf{pdbid} & \textbf{Len} & \textbf{Qubits} & \textbf{Depth} &
\multicolumn{3}{c}{\textbf{Candidate1}} &
\multicolumn{3}{c}{\textbf{Candidate2}} &
\multicolumn{3}{c}{\textbf{Candidate3}} &
\multicolumn{3}{c}{\textbf{Candidate4}} &
\multicolumn{3}{c}{\textbf{Candidate5}} \\
\cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13} \cmidrule(lr){14-16} \cmidrule(lr){17-19}
 &  &  &  &
\textit{rmsd} & \textit{Score} & \textit{Energy} &
\textit{rmsd} & \textit{Score} & \textit{Energy} &
\textit{rmsd} & \textit{Score} & \textit{Energy} &
\textit{rmsd} & \textit{Score} & \textit{Energy} &
\textit{rmsd} & \textit{Score} & \textit{Energy} \\
\midrule
\endhead

\midrule
\multicolumn{19}{r}{\emph{Continued on next page}}\\
\midrule
\endfoot

\bottomrule
\endlastfoot
""".lstrip(
        "\n"
    )

    footer = r"""
\end{longtable}
}
""".lstrip(
        "\n"
    )

    lines = [header]
    for _, group in df.groupby("pdbid", sort=True):
        lines.append(build_row(group))
    lines.append(footer)

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX longtable without inter-column spacing; energies as integers."
    )
    parser.add_argument(
        "--input",
        default="results/quantum_fusion_summary.csv",
        help="Input CSV with columns: pdbid,sequence,length,qubits,depth,cid,energy,Score,rmsd",
    )
    parser.add_argument(
        "--out-tex",
        default="results/table_fused_energy.tex",
        help="Output LaTeX longtable path",
    )
    parser.add_argument(
        "--out-csv",
        default="results/summary_with_energy_int.csv",
        help="Optional output CSV with energy rounded to integer",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input CSV not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)

    required_cols = {
        "pdbid",
        "sequence",
        "length",
        "qubits",
        "depth",
        "cid",
        "energy",
        "Score",
        "rmsd",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns in input CSV: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    df = df[df["cid"].isin(ORDER)].copy()
    # Round energy to integer for all outputs
    df["energy"] = df["energy"].round().astype("Int64")

    # Save integer-energy CSV (optional)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Build LaTeX table
    write_longtable(df, args.out_tex)

    print(f"[OK] Wrote LaTeX longtable to: {args.out_tex}")
    print(f"[OK] Wrote integer-energy CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
