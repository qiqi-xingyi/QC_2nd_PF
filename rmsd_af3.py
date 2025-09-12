# --*-- conding:utf-8 --*--
# @time:9/12/25 00:11
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:rmsd_af3.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch RMSD for AF3 predictions vs dataset fragments (from info.txt / PDBbind).

Input layout (example):
  data/data_set/info.txt
  data/Pdbbind/<pdbid>/<pdbid>_pocket.pdb
  results/af3_out/af3_result/<pdbid>/fold_<pdbid>_model_0.cif ... model_4.cif

Outputs:
  <out_dir>/af3_models.csv   # per (pdb_id, model_k)
  <out_dir>/af3_best.csv     # best model per pdb_id by chosen RMSD metric
"""

import os, re, argparse, warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List

# biopython deps
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from Bio.PDB import PDBParser, MMCIFParser, is_aa
# robust three_to_one across versions
try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    from Bio.Data.IUPACData import protein_letters_3to1 as _3to1
    def three_to_one(resname: str) -> str:
        return _3to1.get(resname.strip().upper(), "X")

# pairwise alignment (keep pairwise2 for simplicity; ok to ignore deprecation)
from Bio import pairwise2

AA1 = set(list("ACDEFGHIKLMNPQRSTVWY"))

# ============== utils: parse info.txt ==============
def parse_info_line(line: str):
    # e.g. 1bai  1bai_pocket.pdb  Chain A  Residues 34-44  ALA-LEU-...
    parts = line.strip().split()
    if not parts or len(parts) < 4:
        return None
    pdb_id = parts[0].lower()
    pocket = parts[1]
    # chain
    if parts[2].lower() == "chain":
        chain = parts[3]; idx = 4
    else:
        chain = parts[2]; idx = 3
    # range
    if idx < len(parts) and parts[idx].lower() == "residues":
        rng = parts[idx+1]
    else:
        rng = parts[idx]
    rng = rng.replace(",", "")
    a, b = rng.split("-")
    start, end = int(a), int(b)
    return dict(pdb_id=pdb_id, pocket=pocket, chain=chain, start=start, end=end)

def load_info(path: str) -> Dict[str, dict]:
    info = {}
    with open(path, "r") as f:
        for ln in f:
            if not ln.strip(): continue
            rec = parse_info_line(ln)
            if rec: info[rec["pdb_id"]] = rec
    return info

# ============== read reference fragment (Cα) ==============
def extract_fragment_ca(pocket_path: str, chain_id: str, start: int, end: int) -> Tuple[str, np.ndarray]:
    parser = MMCIFParser(QUIET=True) if pocket_path.lower().endswith((".cif",".mmcif")) \
        else PDBParser(QUIET=True, PERMISSIVE=True)
    struct = parser.get_structure(os.path.basename(pocket_path), pocket_path)
    model = list(struct.get_models())[0]
    target = None
    for ch in model.get_chains():
        if ch.id.strip() == chain_id.strip():
            target = ch; break
    if target is None:
        raise ValueError(f"Chain {chain_id} not found in {pocket_path}")

    seq1, coords = [], []
    for res in target:
        if not is_aa(res, standard=True): continue
        rno = res.id[1]
        if rno < start or rno > end: continue
        if "CA" not in res: continue
        try:
            one = three_to_one(res.get_resname())
        except KeyError:
            one = "X"
        if one == "X": continue
        seq1.append(one)
        coords.append(res["CA"].coord)

    if not coords:
        raise ValueError(f"No Cα in {pocket_path}:{chain_id} {start}-{end}")
    return "".join(seq1), np.asarray(coords, float)

# ============== AF3 model parsing (choose best chain) ==============
def read_all_chains_ca(pred_path: str) -> List[Tuple[str, str, np.ndarray]]:
    """
    Return list of (chain_id, seq1, ca_xyz) from a predicted structure (.cif/.pdb).
    """
    parser = MMCIFParser(QUIET=True) if pred_path.lower().endswith((".cif",".mmcif")) \
        else PDBParser(QUIET=True, PERMISSIVE=True)
    struct = parser.get_structure(os.path.basename(pred_path), pred_path)
    model = list(struct.get_models())[0]
    out = []
    for ch in model.get_chains():
        seq1, xyz = [], []
        for res in ch:
            if not is_aa(res, standard=True): continue
            if "CA" not in res: continue
            try:
                one = three_to_one(res.get_resname())
            except KeyError:
                one = "X"
            if one == "X": continue
            seq1.append(one)
            xyz.append(res["CA"].coord)
        if seq1 and xyz:
            out.append((ch.id, "".join(seq1), np.asarray(xyz, float)))
    return out

# ============== alignment & RMSD ==============
def align_indices(seq_pred: str, seq_ref: str):
    alns = pairwise2.align.globalms(seq_pred, seq_ref, 1, -1, -5, -1, one_alignment_only=True)
    if not alns:
        return np.array([], int), np.array([], int)
    a_pred, a_ref, *_ = alns[0]
    ip = ir = 0; Ipred=[]; Iref=[]
    for c1, c2 in zip(a_pred, a_ref):
        if c1 != "-" and c2 != "-":
            Ipred.append(ip); Iref.append(ir)
        if c1 != "-": ip += 1
        if c2 != "-": ir += 1
    return np.array(Ipred, int), np.array(Iref, int)

def kabsch_rmsd(P, Q) -> float:
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    Pr = Pc @ R
    return float(np.sqrt(((Pr - Qc)**2).sum(1).mean()))

def similarity_rmsd(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    s = np.trace(np.diag(S)) / (Pc**2).sum() if (Pc**2).sum() > 0 else 1.0
    Pr = s * (Pc @ R)
    return float(np.sqrt(((Pr - Qc)**2).sum(1).mean())), float(s)

# ============== file discovery ==============
def list_af3_models(folder: str, pdb_id: str) -> Dict[str, str]:
    """
    Return mapping 'model_0'.. -> filepath for files like:
      fold_<pdbid>_model_0.cif  (also accept .pdb)
    """
    out = {}
    if not os.path.isdir(folder):
        return out
    pat = re.compile(rf"(?:^|_)model_([0-9]+)\.(?:cif|pdb)$", re.IGNORECASE)
    for nm in os.listdir(folder):
        m = pat.search(nm)
        if m and f"fold_{pdb_id}" in nm.lower():
            out[f"model_{m.group(1)}"] = os.path.join(folder, nm)
    # fallback: accept any *_model_k.{cif,pdb} if fold_<pdbid> prefix not present
    if not out:
        for nm in os.listdir(folder):
            m = pat.search(nm)
            if m:
                out[f"model_{m.group(1)}"] = os.path.join(folder, nm)
    return out

# ============== core per-target compute ==============
def compute_one_af3(pdb_id: str, rec: dict, args, rows_models: list, rows_best: list):
    pocket_path = os.path.join(args.pdbbind_root, pdb_id, rec["pocket"])
    if not os.path.exists(pocket_path):
        alt = os.path.join(args.pdbbind_root, rec["pocket"])
        if os.path.exists(alt): pocket_path = alt
        else: raise FileNotFoundError(f"Pocket file not found: {pocket_path}")

    ref_seq, ref_ca = extract_fragment_ca(pocket_path, rec["chain"], rec["start"], rec["end"])

    af3_dir = os.path.join(args.af3_root, pdb_id)
    models = list_af3_models(af3_dir, pdb_id)
    if not models:
        raise FileNotFoundError(f"No AF3 models found in {af3_dir}")

    best_row = None
    best_metric = None  # choose by rigid or scale depending on args.mode

    for tag, fpath in sorted(models.items(), key=lambda kv: int(kv[0].split("_")[-1])):
        # read all chains and pick the one with max matches to ref fragment
        chains = read_all_chains_ca(fpath)
        if not chains:
            rows_models.append(dict(pdb_id=pdb_id, model=tag, file=os.path.basename(fpath),
                                    error="no_aa_chains"))
            continue

        chosen = None
        chosen_match = -1
        chosen_ip_ir = None

        for ch_id, seq_pred, X_pred in chains:
            ip, ir = align_indices(seq_pred, ref_seq)
            if len(ip) > chosen_match:
                chosen_match = len(ip)
                chosen = (ch_id, seq_pred, X_pred)
                chosen_ip_ir = (ip, ir)

        row = dict(pdb_id=pdb_id, model=tag, file=os.path.basename(fpath),
                   chain_selected=chosen[0] if chosen else None,
                   n_ref=len(ref_seq), n_match=int(chosen_match))

        if chosen and chosen_match >= 3:
            _, seq_pred, X_pred = chosen
            ip, ir = chosen_ip_ir
            P, Q = X_pred[ip], ref_ca[ir]

            if args.mode in ("rigid", "both"):
                r = kabsch_rmsd(P, Q)
                row["rmsd_rigid_A"] = round(r, 3)
            if args.mode in ("scale", "both"):
                r2, s = similarity_rmsd(P, Q)
                row["rmsd_scale_A"] = round(r2, 3)
                row["similarity_scale"] = round(s, 4)
        else:
            row["error"] = "too_few_matches"

        rows_models.append(row)

        # track best per pdb_id
        metric_name = "rmsd_rigid_A" if args.mode in ("rigid", "both") else "rmsd_scale_A"
        if metric_name in row:
            mval = row[metric_name]
            if best_metric is None or mval < best_metric:
                best_metric = mval
                best_row = row

    # best summary
    if best_row is not None:
        rows_best.append(dict(pdb_id=pdb_id, **best_row))
    else:
        rows_best.append(dict(pdb_id=pdb_id, error="no_valid_model"))

# ============== CLI ==============
def main():
    ap = argparse.ArgumentParser(description="RMSD for AF3 predictions vs dataset fragments")
    ap.add_argument("--info", required=True)
    ap.add_argument("--pdbbind-root", required=True)
    ap.add_argument("--af3-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["rigid", "scale", "both"], default="rigid")
    # default args for IDE-run convenience (optional)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    info = load_info(args.info)

    rows_models, rows_best = [], []

    for pdb_id in sorted(os.listdir(args.af3_root)):
        pid = pdb_id.lower()
        if pid not in info:
            continue
        try:
            compute_one_af3(pid, info[pid], args, rows_models, rows_best)
            print(f"[OK] {pid}")
        except Exception as e:
            print(f"[ERR] {pid}: {e}")
            rows_best.append(dict(pdb_id=pid, error=str(e)))

    f_models = os.path.join(args.out_dir, "af3_models.csv")
    f_best = os.path.join(args.out_dir, "af3_best.csv")
    pd.DataFrame(rows_models).to_csv(f_models, index=False)
    pd.DataFrame(rows_best).to_csv(f_best, index=False)
    print(f"Saved: {f_models}")
    print(f"Saved: {f_best}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv += [
            "--info", "data/data_set/info.txt",
            "--pdbbind-root", "data/Pdbbind",
            "--af3-root", "results/af3_out/af3_result",
            "--out-dir", "results/af3_rmsd",
            "--mode", "rigid"
        ]
    main()
