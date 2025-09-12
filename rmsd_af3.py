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
  data/Pdbbind/<pdbid>/<pdbid>_protein.pdb   # preferred if exists
  data/Pdbbind/<pdbid>/<pdbid>_pocket.pdb    # fallback
  results/af3_out/af3_result/<pdbid>/fold_<pdbid>_model_0.cif ... model_4.cif

Outputs:
  <out_dir>/af3_models.csv   # per (pdb_id, model_k)
  <out_dir>/af3_best.csv     # best model per pdb_id by chosen metric
"""

import os
import re
import argparse
import warnings
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# --- Biopython imports and compatibility helpers ---
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from Bio.PDB import PDBParser, MMCIFParser, is_aa

# robust three_to_one across versions
try:
    from Bio.PDB.Polypeptide import three_to_one
except Exception:
    from Bio.Data.IUPACData import protein_letters_3to1 as _3to1
    def three_to_one(resname: str) -> str:
        return _3to1.get(resname.strip().upper(), "X")

# pairwise alignment (pairwise2 is deprecated but OK to use here)
from Bio import pairwise2
from Bio.Data.IUPACData import protein_letters_3to1 as _3to1_map

AA1 = set(list("ACDEFGHIKLMNPQRSTVWY"))

# ----------------------------- info.txt parsing -----------------------------
def tri_to_one(seq3: str) -> str:
    """Convert tri-letter residues like 'GLY-ASN-ASP' to one-letter string."""
    toks = re.split(r"[-\s]+", seq3.strip())
    out = []
    for t in toks:
        if not t:
            continue
        out.append(_3to1_map.get(t.upper(), "X"))
    return "".join(out)

def parse_info_line(line: str):
    # example:
    # 1bai 1bai_pocket.pdb Chain A Residues 34-44 ALA-LEU-...
    parts = line.strip().split()
    if not parts or len(parts) < 4:
        return None
    pdb_id = parts[0].lower()
    pocket = parts[1]
    # chain
    if parts[2].lower() == "chain":
        chain = parts[3]
        idx = 4
    else:
        chain = parts[2]
        idx = 3
    # range
    if idx < len(parts) and parts[idx].lower() == "residues":
        rng = parts[idx + 1]
        seq_tokens = parts[idx + 2:]
    else:
        rng = parts[idx]
        seq_tokens = parts[idx + 1:]
    rng = rng.replace(",", "")
    a, b = rng.split("-")
    start, end = int(a), int(b)
    seq3 = " ".join(seq_tokens)
    seq1 = tri_to_one(seq3) if seq3 else ""
    return dict(pdb_id=pdb_id, pocket=pocket, chain=chain, start=start, end=end, seq1=seq1)

def load_info(path: str) -> Dict[str, dict]:
    info = {}
    with open(path, "r") as f:
        for ln in f:
            if not ln.strip():
                continue
            rec = parse_info_line(ln)
            if rec:
                info[rec["pdb_id"]] = rec
    return info

# ----------------------------- reference selection -----------------------------
def find_ref_pdb(pdbbind_root: str, pdb_id: str, pocket_name: str) -> str:
    """
    Prefer <pdbid>_protein.pdb if present (keeps original numbering),
    otherwise fall back to <pdbid>_pocket.pdb. Also try directly under root.
    """
    candidates = [
        os.path.join(pdbbind_root, pdb_id, f"{pdb_id}_protein.pdb"),
        os.path.join(pdbbind_root, pdb_id, pocket_name),
        os.path.join(pdbbind_root, f"{pdb_id}_protein.pdb"),
        os.path.join(pdbbind_root, pocket_name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Reference PDB not found for {pdb_id}: tried protein & pocket")

def chain_seq_ca(chain):
    """Return (seq1, CA_xyz, res_ids) for a chain."""
    seq1, xyz, ridx = [], [], []
    for res in chain:
        if not is_aa(res, standard=True):
            continue
        if "CA" not in res:
            continue
        try:
            one = three_to_one(res.get_resname())
        except KeyError:
            one = "X"
        if one == "X":
            continue
        seq1.append(one)
        xyz.append(res["CA"].coord)
        ridx.append(res.id)  # (hetflag, resseq, icode)
    return "".join(seq1), np.asarray(xyz, float), ridx

def extract_fragment_ca_robust(ref_path: str,
                               chain_id: str,
                               start: int,
                               end: int,
                               seq_from_info: Optional[str] = None) -> Tuple[str, np.ndarray]:
    """
    Try to extract the fragment by residue numbering first.
    If that fails (common in pocket files), fall back to sequence-window matching.
    """
    parser = MMCIFParser(QUIET=True) if ref_path.lower().endswith((".cif", ".mmcif")) \
        else PDBParser(QUIET=True, PERMISSIVE=True)
    struct = parser.get_structure(os.path.basename(ref_path), ref_path)
    model = list(struct.get_models())[0]

    # locate the requested chain; fallback to longest AA chain
    target = None
    for ch in model.get_chains():
        if ch.id.strip() == chain_id.strip():
            target = ch
            break
    if target is None:
        target = max(model.get_chains(),
                     key=lambda ch: sum(1 for r in ch if is_aa(r, standard=True)))

    # (i) try numbering window
    seq_num, xyz_num = [], []
    for res in target:
        if not is_aa(res, standard=True):
            continue
        rno = res.id[1]
        if start <= rno <= end and "CA" in res:
            try:
                one = three_to_one(res.get_resname())
            except KeyError:
                one = "X"
            if one != "X":
                seq_num.append(one)
                xyz_num.append(res["CA"].coord)
    if xyz_num:
        return "".join(seq_num), np.asarray(xyz_num, float)

    # (ii) local sequence-window alignment
    full_seq, full_xyz, _ = chain_seq_ca(target)
    if seq_from_info and len(full_seq) >= 3:
        alns = pairwise2.align.localms(full_seq, seq_from_info, 2, -1, -5, -1, one_alignment_only=True)
        if alns:
            a_full, a_info, *_ = alns[0]
            i_full = j_info = 0
            idx_full = []
            for c1, c2 in zip(a_full, a_info):
                if c1 != "-" and c2 != "-":
                    idx_full.append(i_full)
                if c1 != "-":
                    i_full += 1
                if c2 != "-":
                    j_info += 1
            if len(idx_full) >= 3:
                return "".join(full_seq[i] for i in idx_full), full_xyz[idx_full]

    # (iii) global alignment fallback
    if seq_from_info and len(full_seq) >= 3:
        alns = pairwise2.align.globalms(full_seq, seq_from_info, 1, -1, -5, -1, one_alignment_only=True)
        if alns:
            a_full, a_info, *_ = alns[0]
            i_full = j_info = 0
            idx_full = []
            for c1, c2 in zip(a_full, a_info):
                if c1 != "-" and c2 != "-":
                    idx_full.append(i_full)
                if c1 != "-":
                    i_full += 1
                if c2 != "-":
                    j_info += 1
            if len(idx_full) >= 3:
                return "".join(full_seq[i] for i in idx_full), full_xyz[idx_full]

    raise ValueError(f"No CA fragment found via numbering or sequence window in {ref_path}:{chain_id} {start}-{end}")

# ----------------------------- alignment & RMSD -----------------------------
def align_indices(seq_pred: str, seq_ref: str):
    alns = pairwise2.align.globalms(seq_pred, seq_ref, 1, -1, -5, -1, one_alignment_only=True)
    if not alns:
        return np.array([], int), np.array([], int)
    a_pred, a_ref, *_ = alns[0]
    ip = ir = 0
    Ipred, Iref = [], []
    for c1, c2 in zip(a_pred, a_ref):
        if c1 != "-" and c2 != "-":
            Ipred.append(ip)
            Iref.append(ir)
        if c1 != "-":
            ip += 1
        if c2 != "-":
            ir += 1
    return np.array(Ipred, int), np.array(Iref, int)

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Pr = Pc @ R
    return float(np.sqrt(((Pr - Qc) ** 2).sum(1).mean()))

def similarity_rmsd(P: np.ndarray, Q: np.ndarray) -> Tuple[float, float]:
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    s = np.trace(np.diag(S)) / (Pc ** 2).sum() if (Pc ** 2).sum() > 0 else 1.0
    Pr = s * (Pc @ R)
    return float(np.sqrt(((Pr - Qc) ** 2).sum(1).mean())), float(s)

# ----------------------------- AF3 model discovery -----------------------------
def list_af3_models(folder: str, pdb_id: str) -> Dict[str, str]:
    """
    Return mapping 'model_0'.. -> filepath for files like:
      fold_<pdbid>_model_0.cif  (also accept .pdb)
    If the prefix is missing, any *_model_k.{cif,pdb} is accepted.
    """
    out = {}
    if not os.path.isdir(folder):
        return out
    pat = re.compile(r"(?:^|_)model_([0-9]+)\.(?:cif|pdb)$", re.IGNORECASE)
    for nm in os.listdir(folder):
        m = pat.search(nm)
        if m and f"fold_{pdb_id}" in nm.lower():
            out[f"model_{m.group(1)}"] = os.path.join(folder, nm)
    if not out:
        for nm in os.listdir(folder):
            m = pat.search(nm)
            if m:
                out[f"model_{m.group(1)}"] = os.path.join(folder, nm)
    return out

# ----------------------------- per-target compute -----------------------------
def compute_one_af3(pdb_id: str, rec: dict, args, rows_models: list, rows_best: list):
    ref_path = find_ref_pdb(args.pdbbind_root, pdb_id, rec["pocket"])
    ref_seq, ref_ca = extract_fragment_ca_robust(ref_path, rec["chain"], rec["start"], rec["end"], rec.get("seq1", ""))

    af3_dir = os.path.join(args.af3_root, pdb_id)
    models = list_af3_models(af3_dir, pdb_id)
    if not models:
        raise FileNotFoundError(f"No AF3 models found in {af3_dir}")

    best_row = None
    best_metric = None
    metric_name = "rmsd_rigid_A" if args.mode in ("rigid", "both") else "rmsd_scale_A"

    for tag, fpath in sorted(models.items(), key=lambda kv: int(kv[0].split("_")[-1])):
        parser = MMCIFParser(QUIET=True) if fpath.lower().endswith((".cif", ".mmcif")) \
            else PDBParser(QUIET=True, PERMISSIVE=True)
        struct = parser.get_structure(os.path.basename(fpath), fpath)
        model0 = list(struct.get_models())[0]

        # choose chain with max matched residues against ref fragment
        chosen = None
        chosen_match = -1
        chosen_ip_ir = None

        for ch in model0.get_chains():
            seq1, xyz, _ = chain_seq_ca(ch)
            if not seq1:
                continue
            ip, ir = align_indices(seq1, ref_seq)
            if len(ip) > chosen_match:
                chosen_match = len(ip)
                chosen = (ch.id, seq1, xyz)
                chosen_ip_ir = (ip, ir)

        row = dict(pdb_id=pdb_id, model=tag, file=os.path.basename(fpath),
                   chain_selected=chosen[0] if chosen else None,
                   n_ref=len(ref_seq), n_match=int(chosen_match if chosen_match >= 0 else 0))

        if chosen and chosen_match >= 3:
            _, seq_pred, X_pred = chosen
            ip, ir = chosen_ip_ir
            P, Q = X_pred[ip], ref_ca[ir]
            if args.mode in ("rigid", "both"):
                row["rmsd_rigid_A"] = round(kabsch_rmsd(P, Q), 3)
            if args.mode in ("scale", "both"):
                r2, s = similarity_rmsd(P, Q)
                row["rmsd_scale_A"] = round(r2, 3)
                row["similarity_scale"] = round(s, 4)
        else:
            row["error"] = "too_few_matches"

        rows_models.append(row)

        if metric_name in row:
            mval = row[metric_name]
            if best_metric is None or mval < best_metric:
                best_metric = mval
                best_row = row

    rows_best.append(dict(pdb_id=pdb_id, **(best_row if best_row else {"error": "no_valid_model"})))

# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description="RMSD for AF3 predictions vs dataset fragments")
    ap.add_argument("--info", required=True)
    ap.add_argument("--pdbbind-root", required=True)
    ap.add_argument("--af3-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["rigid", "scale", "both"], default="rigid")
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
    # Convenience defaults for IDE runs: supply args if none are given.
    import sys
    if len(sys.argv) == 1:
        sys.argv += [
            "--info", "data/data_set/info.txt",
            "--pdbbind-root", "data/Pdbbind",
            "--af3-root", "results/af3_out/af3_result",
            "--out-dir", "results/af3_rmsd",
            "--mode", "rigid",
        ]
    main()

