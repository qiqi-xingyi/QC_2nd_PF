# --*-- conding:utf-8 --*--
# @time:9/21/25 02:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:get_quantum_info.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd


seqs_file = "data/seqs_len10.csv"
quantum_dir = "data/Quantum_original_data"
output_file = "data/quantum_summary.csv"


resource_map = {
    14: (102, 413),
    13: (92, 373),
    12: (82, 333),
    11: (72, 293),
    10: (63, 257),
}

seqs_df = pd.read_csv(seqs_file)


records = []

if __name__ == '__main__':

    for _, row in seqs_df.iterrows():
        pdbid = row["id"]
        sequence = row["sequence"]
        length = len(sequence)


        qubits, depth = resource_map.get(length, (None, None))

        energy_file = os.path.join(quantum_dir, pdbid, f"top_5_energies_{pdbid}.txt")
        energies = [None] * 5

        if os.path.exists(energy_file):
            with open(energy_file, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:5]):
                    try:

                        value = float(line.strip().split(":")[1])
                        energies[i] = value
                    except Exception:
                        continue
        else:
            print(f"[WARN] Energy file not found for {pdbid}")

        records.append({
            "pdbid": pdbid,
            "sequence": sequence,
            "length": length,
            "qubits": qubits,
            "depth": depth,
            "energy_rank1": energies[0],
            "energy_rank2": energies[1],
            "energy_rank3": energies[2],
            "energy_rank4": energies[3],
            "energy_rank5": energies[4],
        })


    df_out = pd.DataFrame(records)
    df_out.to_csv(output_file, index=False)

    print(f"[OK] Wrote summary to: {output_file}")
