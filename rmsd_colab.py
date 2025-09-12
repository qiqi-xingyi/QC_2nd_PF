# --*-- conding:utf-8 --*--
# @time:9/12/25 01:15
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:rmsd_colab.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute RMSD of ColabFold predicted structures vs reference dataset windows.

Input:
  - info.txt (defines pdb_id, chain, residue range, reference sequence)
  - PDBbind dataset (contains reference protein structures)
  - ColabFold outputs in results/colabfold_out/colabfold_result/<subdir>/
    where subdir starts with pdb_id (first 4 letters)
    and contains files *_unrelaxed_rank_*.pdb

Output:
  - <out_dir>/colabfold_models.csv   # RMSD per model
  - <out_dir>/colabfold_best.csv     # best RMSD per target
"""

import os
import re
import argparse
import warnings
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from Bio.PDB import PDBParser
try:
    from Bio.PDB.Polypeptide import three_to_one
except Exception:
    from Bio.Data.IUPACData import protein_letters_3to1 as _3to1
    def three_to_one(resname: str) -> str:
        return _3to1.get(resname.strip().upper(), "X")

from Bio import pairwise2
from Bio.Data.IUPACData import protein_letters_3to1 as MAP_3TO1

# ---------------- residue helpers ----------------
RESNAME_NORMALIZE = {
    "MSE": "MET", "SEC": "CYS", "PYL": "LYS",
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "SEP": "SER", "TPO": "THR", "PTR": "TYR",
    "CSO": "CYS", "CME": "CYS", "MLY": "LYS",
    "GLX": "GLU", "ASX": "ASP",
}

def normalize_resname(resname: str) -> str:
    rn = resname.strip().upper()
    return RESNAME_NORMALIZE.get(rn, rn)

def chain_to_seq_ca(chain):
    seq1, xyz, resnums = [], [], []
    for res in chain:
        if "CA" not in res:
            continue
        rn = normalize_resname(res.get_resname())
        try:
            one = three_to_one(rn)
        except Exception:
            one = "X"
        seq1.append(one)
        xyz.append(res["CA"].coord)
        resnums.append(res.id[1])
    return "".join(seq1), np.asarray(xyz, float), np.asarray(resnums, int)

# ---------------- info.txt ----------------
def tri_to_one(seq3: str) -> str:
    toks = re.split(r"[-\s]+", seq3.strip())
    return "".join(MAP_3TO1.get(t.upper(), "X") for t in toks if t)

def parse_info_line(line: str):
    parts = line.strip().split()
    if not parts or len(parts) < 4:
        return None
    pdb_id = parts[0].lower()
    if parts[2].lower() == "chain":
        chain = parts[3]; idx = 4
    else:
        chain = parts[2]; idx = 3
    rng = parts[idx+1] if parts[idx].lower() == "residues" else parts[idx]
    a, b = rng.split("-"); start, end = int(a), int(b)
    seq_tokens = parts[idx+2:] if parts[idx].lower() == "residues" else parts[idx+1:]
    seq1 = tri_to_one(" ".join(seq_tokens)) if seq_tokens else ""
    return dict(pdb_id=pdb_id, chain=chain, start=start, end=end, seq1=seq1)

def load_info(path: str) -> Dict[str, dict]:
    info = {}
    with open(path, "r") as f:
        for ln in f:
            rec = parse_info_line(ln)
            if rec:
                info[rec["pdb_id"]] = rec
    return info

# ---------------- RMSD utils ----------------
def align_local_indices(a: str, b: str):
    alns = pairwise2.align.localms(a, b, 2, -1, -5, -1, one_alignment_only=True)
    if not alns:
        return np.array([], int), np.array([], int)
    A, B, *_ = alns[0]
    ia = ib = 0
    Ia, Ib = [], []
    for c1, c2 in zip(A, B):
        if c1 != "-" and c2 != "-":
            Ia.append(ia); Ib.append(ib)
        if c1 != "-": ia += 1
        if c2 != "-": ib += 1
    return np.array(Ia, int), np.array(Ib, int)

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Pr = Pc @ R
    return float(np.sqrt(((Pr - Qc) ** 2).sum(1).mean()))

# ---------------- main compute ----------------
def compute_colabfold_rmsd(pdb_id: str, rec: dict, pdbbind_root: str, colabfold_root: str, rows: list):
    # reference structure
    ref_path = os.path.join(pdbbind_root, pdb_id, f"{pdb_id}_protein.pdb")
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    struct = parser.get_structure(pdb_id, ref_path)
    model = list(struct.get_models())[0]
    ref_chain = [c for c in model.get_chains() if c.id == rec["chain"]][0]
    ref_seq, ref_xyz, ref_resnums = chain_to_seq_ca(ref_chain)
    mask = (ref_resnums >= rec["start"]) & (ref_resnums <= rec["end"])
    ref_seq_win, ref_xyz_win = ref_seq[mask], ref_xyz[mask]

    # colabfold predicted models
    subdirs = [d for d in os.listdir(colabfold_root) if d.startswith(pdb_id)]
    if not subdirs:
        rows.append(dict(pdb_id=pdb_id, error="no_colabfold_dir"))
        return
    pred_dir = os.path.join(colabfold_root, subdirs[0])
    for nm in os.listdir(pred_dir):
        if not nm.endswith(".pdb") or "unrelaxed_rank" not in nm:
            continue
        fpath = os.path.join(pred_dir, nm)
        struct_pred = parser.get_structure(nm, fpath)
        model_pred = list(struct_pred.get_models())[0]
        pred_chain = list(model_pred.get_chains())[0]
        pred_seq, pred_xyz, _ = chain_to_seq_ca(pred_chain)

        ip, ir = align_local_indices(pred_seq, ref_seq)
        if len(ip) < 3:
            rows.append(dict(pdb_id=pdb_id, model=nm, n_match=len(ip), error="no_valid_alignment"))
            continue
        P, Q = pred_xyz[ip], ref_xyz[ir]
        rmsd_val = round(kabsch_rmsd(P, Q), 3)
        rows.append(dict(pdb_id=pdb_id, model=nm, n_match=len(ip), rmsd_A=rmsd_val))

def main():
    ap = argparse.ArgumentParser(description="ColabFold RMSD vs reference dataset")
    ap.add_argument("--info", required=True)
    ap.add_argument("--pdbbind-root", required=True)
    ap.add_argument("--colabfold-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    info = load_info(args.info)
    rows = []
    for sub in sorted(os.listdir(args.colabfold_root)):
        pdb_id = sub[:4].lower()
        if pdb_id not in info:
            continue
        try:
            compute_colabfold_rmsd(pdb_id, info[pdb_id], args.pdbbind_root, args.colabfold_root, rows)
            print(f"[OK] {pdb_id}")
        except Exception as e:
            print(f"[ERR] {pdb_id}: {e}")
            rows.append(dict(pdb_id=pdb_id, error=str(e)))

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv += [
            "--info", "data/data_set/info.txt",
            "--pdbbind-root", "data/Pdbbind",
            "--colabfold-root", "results/colabfold_out/colabfold_result",
            "--out", "results/colabfold_rmsd.csv"
        ]
    main()
