# --*-- conding:utf-8 --*--
# @time:9/21/25 02:28
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:summary.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

if __name__ == '__main__':


    quantum_summary = "data/quantum_summary.csv"
    fusion_file = "results/rmsd_figures/summary_all_candidates.csv"
    output_file = "results/quantum_fusion_summary.csv"


    df_quantum = pd.read_csv(quantum_summary)
    df_fusion = pd.read_csv(fusion_file)


    energy_cols = {
        "top_1": "energy_rank1",
        "top_2": "energy_rank2",
        "top_3": "energy_rank3",
        "top_4": "energy_rank4",
        "top_5": "energy_rank5",
    }

    records = []

    for _, row in df_fusion.iterrows():
        pdbid = row["pdb_id"]
        cid = row["cid"]   # top_1, top_2, ...
        score = row["Score"]
        rmsd = row["rmsd"]


        qrow = df_quantum[df_quantum["pdbid"] == pdbid].iloc[0]

        sequence = qrow["sequence"]
        length = qrow["length"]
        qubits = qrow["qubits"]
        depth = qrow["depth"]


    energy_col = energy_cols[cid]
    energy_val = qrow[energy_col]

    if pd.notna(energy_val):
        energy_val = round(float(energy_val), 2)

    records.append({
        "pdbid": pdbid,
        "sequence": sequence,
        "length": length,
        "qubits": qubits,
        "depth": depth,
        "cid": cid,
        "energy": energy_val,
        "Score": score,
        "rmsd": rmsd,
    })


df_out = pd.DataFrame(records)
df_out.to_csv(output_file, index=False)

print(f"[OK] Wrote merged summary to: {output_file}")
