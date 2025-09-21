# --*-- conding:utf-8 --*--
# @time:9/21/25 02:39
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:get_tex_table.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from collections import OrderedDict

if __name__ == '__main__':

    inp = "results/quantum_fusion_summary.csv"
    out = "results/table_fused_energy.tex"

    df = pd.read_csv(inp)

    order = ["top_1","top_2","top_3","top_4","top_5"]
    df = df[df["cid"].isin(order)]
    df["cid"] = pd.Categorical(df["cid"], categories=order, ordered=True)
    df = df.sort_values(["pdbid","cid"])

    rows = []
    current = None
    for pdbid, g in df.groupby("pdbid"):
        g = g.set_index("cid").loc[order]
        length = int(g["length"].iloc[0])
        qubits = int(g["qubits"].iloc[0])
        depth  = int(g["depth"].iloc[0])
        # 拼一行：pdbid & len & qubits & depth & (rmsd,Score,Energy)*5
        cells = [pdbid, str(length), str(qubits), str(depth)]
        for cid in order:
            r = g.loc[cid]
            rmsd = f"{float(r['rmsd']):.2f}".rstrip('0').rstrip('.') if '.' in f"{float(r['rmsd']):.2f}" else f"{float(r['rmsd']):.2f}"
            score= f"{float(r['Score']):.2f}".rstrip('0').rstrip('.') if '.' in f"{float(r['Score']):.2f}" else f"{float(r['Score']):.2f}"
            energy = f"{float(r['energy']):.2f}"
            cells += [rmsd, score, energy]
        # 用 LaTeX 的行
        row_tex = " & ".join(cells) + r" \\"
        rows.append(row_tex)

    # 头尾模板
    header = r"""
    {\scriptsize
    \setlength{\tabcolsep}{3pt}
    \renewcommand{\arraystretch}{0.9}
    \begin{longtable}{lrrrr
      @{\hspace{20pt}}rrr
      @{\hspace{28pt}}rrr
      @{\hspace{28pt}}rrr
      @{\hspace{28pt}}rrr
      @{\hspace{28pt}}rrr}
    \caption{Per-fragment summary with fused score (Score), RMSD, and quantum energy for the top-5 candidates. Left block reports chain length, qubit count, and circuit depth; each candidate block lists \textit{rmsd}, \textit{Score}, and \textit{Energy}.}
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
    """
    footer = r"""
    \end{longtable}
    }
    """

    with open(out, "w") as f:
        f.write(header)
        for r in rows:
            f.write(r + "\n")
        f.write(footer)

    print(f"[OK] Wrote LaTeX longtable to: {out}")
