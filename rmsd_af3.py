# --*-- conding:utf-8 --*--
# @time:9/12/25 00:11
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:rmsd_af3.py


"""
AF3 short-fragment RMSD vs dataset fragments.

AF3 predicts short chains; reference files are full proteins.
We align the AF3 chain sequence to the full reference chain sequence, then
restrict the matched positions to those whose reference residue numbers fall
inside the (start, end) window from info.txt, and compute RMSD.

Inputs (example):
  --info data/data_set/info.txt
  --pdbbind-root data/Pdbbind
  --af3-root results/af3_out/af3_result

Output:
  --out-dir results/af3_rmsd
    - af3_models.csv   (per model_k)
    - af3_best.csv     (best per target by chosen metric)
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

from Bio.PDB import PDBParser, MMCIFParser, is_aa
try:
    from Bio.PDB.Polypeptide import three_to_one
except Exception:
    from Bio.Data.IUPACData import protein_letters_3to1 as _3to1
    def three_to_one(resname: str) -> str:
        return _3to1.get(resname.strip().upper(), "X")

from Bio import pairwise2
from Bio.Data.IUPACData import protein_letters_3to1 as _3to1_map

# -------------------- info.txt parsing --------------------
def tri_to_one(seq3: str) -> str:
    toks = re.split(r"[-\s]+", seq3.strip())
    return "".join(_3to1_map.get(t.upper(), "X") for t in toks if t)

def parse_info_line(line: str):
    # 1bai 1bai_pocket.pdb Chain A Residues 34-44 ALA-LEU-...
    parts = line.strip().split()
    if not parts or len(parts) < 4:
        return None
    pdb_id = parts[0].lower()
    pocket = parts[1]
    if parts[2].lower() == "chain":
        chain = parts[3]; idx = 4
    else:
        chain = parts[2]; idx = 3
    if idx < len(parts) and parts[idx].lower() == "residues":
        rng = parts[idx + 1]
        seq_tokens = parts[idx + 2:]
    else:
        rng = parts[idx]
        seq_tokens = parts[idx + 1:]
    rng = rng.replace(",", "")
    a, b = rng.split("-")
    start, end = int(a), int(b)
    seq1 = tri_to_one(" ".join(seq_tokens)) if seq_tokens else ""
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

# -------------------- reference selection --------------------
def find_ref_protein(pdbbind_root: str, pdb_id: str, pocket_name: str) -> str:
    # Prefer <id>_protein.pdb, fall back to pocket, also try directly under root
    candidates = [
        os.path.join(pdbbind_root, pdb_id, f"{pdb_id}_protein.pdb"),
        os.path.join(pdbbind_root, pdb_id, pocket_name),
        os.path.join(pdbbind_root, f"{pdb_id}_protein.pdb"),
        os.path.join(pdbbind_root, pocket_name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Reference PDB not found for {pdb_id}")

# -------------------- structure helpers --------------------
def extract_chain_full(ref_path: str, chain_id: str):
    """Return the entire reference chain as (seq1, CA_xyz, resnums)."""
    parser = MMCIFParser(QUIET=True) if ref_path.lower().endswith((".cif", ".mmcif")) \
        else PDBParser(QUIET=True, PERMISSIVE=True)
    struct = parser.get_structure(os.path.basename(ref_path), ref_path)
    model = list(struct.get_models())[0]

    target = None
    for ch in model.get_chains():
        if ch.id.strip() == chain_id.strip():
            target = ch
            break
    if target is None:
        # fallback: longest AA chain
        target = max(model.get_chains(),
                     key=lambda ch: sum(1 for r in ch if is_aa(r, standard=True)))

    seq1, xyz, resnums = [], [], []
    for res in target:
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
        # residue number (resseq); ignore insertion code for window filtering
        resnums.append(res.id[1])
    if not seq1:
        raise ValueError(f"No CA residues found in reference chain of {ref_path}")
    return "".join(seq1), np.asarray(xyz, float), np.asarray(resnums, int)

def read_af3_models(af3_dir: str, pdb_id: str) -> Dict[str, str]:
    """Map model tag -> file path for fold_<id>_model_k.(cif|pdb) or *_model_k.*"""
    out = {}
    if not os.path.isdir(af3_dir):
        return out
    pat = re.compile(r"(?:^|_)model_([0-9]+)\.(?:cif|pdb)$", re.IGNORECASE)
    for nm in os.listdir(af3_dir):
        if f"fold_{pdb_id}" in nm.lower():
            m = pat.search(nm)
            if m:
                out[f"model_{m.group(1)}"] = os.path.join(af3_dir, nm)
    if not out:
        for nm in os.listdir(af3_dir):
            m = pat.search(nm)
            if m:
                out[f"model_{m.group(1)}"] = os.path.join(af3_dir, nm)
    return out

def read_all_pred_chains(pred_path: str):
    """Return list of (chain_id, seq1, CA_xyz) from AF3 predicted structure."""
    parser = MMCIFParser(QUIET=True) if pred_path.lower().endswith((".cif", ".mmcif")) \
        else PDBParser(QUIET=True, PERMISSIVE=True)
    struct = parser.get_structure(os.path.basename(pred_path), pred_path)
    model = list(struct.get_models())[0]
    chains = []
    for ch in model.get_chains():
        seq1, xyz = [], []
        for res in ch:
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
        if seq1 and xyz:
            chains.append((ch.id, "".join(seq1), np.asarray(xyz, float)))
    return chains

# -------------------- alignment & RMSD --------------------
def align_global_indices(seq_a: str, seq_b: str):
    """Global alignment indices mapping between seq_a and seq_b."""
    alns = pairwise2.align.globalms(seq_a, seq_b, 1, -1, -5, -1, one_alignment_only=True)
    if not alns:
        return np.array([], int), np.array([], int)
    a, b, *_ = alns[0]
    ia = ib = 0
    Ia, Ib = [], []
    for c1, c2 in zip(a, b):
        if c1 != "-" and c2 != "-":
            Ia.append(ia); Ib.append(ib)
        if c1 != "-": ia += 1
        if c2 != "-": ib += 1
    return np.array(Ia, int), np.array(Ib, int)

def align_local_indices(seq_a: str, seq_b: str):
    """Local alignment indices mapping between seq_a and seq_b."""
    alns = pairwise2.align.localms(seq_a, seq_b, 2, -1, -5, -1, one_alignment_only=True)
    if not alns:
        return np.array([], int), np.array([], int)
    a, b, *_ = alns[0]
    ia = ib = 0
    Ia, Ib = [], []
    for c1, c2 in zip(a, b):
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

def similarity_rmsd(P: np.ndarray, Q: np.ndarray):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    s = np.trace(np.diag(S)) / (Pc ** 2).sum() if (Pc ** 2).sum() > 0 else 1.0
    Pr = s * (Pc @ R)
    return float(np.sqrt(((Pr - Qc) ** 2).sum(1).mean())), float(s)

# -------------------- core compute --------------------
def compute_one(pdb_id: str, rec: dict, args, rows_models: list, rows_best: list):
    # load reference full chain
    ref_path = find_ref_protein(args.pdbbind_root, pdb_id, rec["pocket"])
    ref_seq_full, ref_xyz_full, ref_resnums = extract_chain_full(ref_path, rec["chain"])

    # enumerate AF3 models
    af3_dir = os.path.join(args.af3_root, pdb_id)
    models = read_af3_models(af3_dir, pdb_id)
    if not models:
        raise FileNotFoundError(f"No AF3 models in {af3_dir}")

    best_row = None
    best_metric = None
    metric_name = "rmsd_rigid_A" if args.mode in ("rigid", "both") else "rmsd_scale_A"

    for tag, fpath in sorted(models.items(), key=lambda kv: int(kv[0].split("_")[-1])):
        chains = read_all_pred_chains(fpath)
        if not chains:
            rows_models.append(dict(pdb_id=pdb_id, model=tag, file=os.path.basename(fpath),
                                    error="no_pred_chain"))
            continue

        # choose chain with max overlap inside the desired ref window [start,end]
        chosen = None
        chosen_overlap = -1
        chosen_ip = chosen_ir = None

        for ch_id, seq_pred, xyz_pred in chains:
            # local alignment AF3 (pred) vs reference full chain
            ip_full, ir_full = align_local_indices(seq_pred, ref_seq_full)
            if len(ip_full) < 3:
                continue
            # filter to reference residues within [start, end]
            mask = (ref_resnums[ir_full] >= rec["start"]) & (ref_resnums[ir_full] <= rec["end"])
            if mask.sum() < 3:
                continue
            ip, ir = ip_full[mask], ir_full[mask]
            overlap = len(ip)
            if overlap > chosen_overlap:
                chosen_overlap = overlap
                chosen = (ch_id, seq_pred, xyz_pred)
                chosen_ip, chosen_ir = ip, ir

        row = dict(pdb_id=pdb_id, model=tag, file=os.path.basename(fpath),
                   chain_selected=chosen[0] if chosen else None,
                   n_ref_window=rec["end"] - rec["start"] + 1,
                   n_match=int(chosen_overlap if chosen_overlap >= 0 else 0))

        if chosen and chosen_overlap >= 3:
            _, seq_pred, X_pred = chosen
            P = X_pred[chosen_ip]
            Q = ref_xyz_full[chosen_ir]
            if args.mode in ("rigid", "both"):
                row["rmsd_rigid_A"] = round(kabsch_rmsd(P, Q), 3)
            if args.mode in ("scale", "both"):
                r2, s = similarity_rmsd(P, Q)
                row["rmsd_scale_A"] = round(r2, 3)
                row["similarity_scale"] = round(s, 4)
        else:
            row["error"] = "no_overlap_in_window"

        rows_models.append(row)

        if metric_name in row:
            m = row[metric_name]
            if best_metric is None or m < best_metric:
                best_metric = m
                best_row = row

    rows_best.append(dict(pdb_id=pdb_id, **(best_row if best_row else {"error": "no_valid_model"})))

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="AF3 short-fragment RMSD vs dataset window")
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
            compute_one(pid, info[pid], args, rows_models, rows_best)
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
            "--mode", "rigid",
        ]
    main()


