# --*-- conding:utf-8 --*--
# @time:9/12/25 00:11
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:rmsd_af3.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AF3 short-fragment RMSD vs dataset reference window.

Workflow:
1) Read info.txt (pdb_id, pocket file, chain, start-end, optional tri-letter seq).
2) Prefer <id>_protein.pdb as reference (fallback to pocket if missing).
3) Extract the best reference chain (try requested chain; fallback to the most relevant chain).
4) For each AF3 model: read predicted chain(s), do local alignment to the full reference chain,
   then keep only matched positions whose reference residue numbers fall in [start, end].
5) Compute RMSD (rigid; optional similarity RMSD) on those positions.

Outputs:
  <out_dir>/af3_models.csv
  <out_dir>/af3_best.csv
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

# three_to_one compatibility across Biopython versions
try:
    from Bio.PDB.Polypeptide import three_to_one
except Exception:
    from Bio.Data.IUPACData import protein_letters_3to1 as _3to1
    def three_to_one(resname: str) -> str:
        return _3to1.get(resname.strip().upper(), "X")

from Bio import pairwise2
from Bio.Data.IUPACData import protein_letters_3to1 as MAP_3TO1


# -------------------- info.txt parsing --------------------
def tri_to_one(seq3: str) -> str:
    toks = re.split(r"[-\s]+", seq3.strip())
    return "".join(MAP_3TO1.get(t.upper(), "X") for t in toks if t)

def parse_info_line(line: str):
    # example: 1bai 1bai_pocket.pdb Chain A Residues 34-44 ALA-LEU-...
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


# -------------------- reference file selection --------------------
def find_reference_protein(pdbbind_root: str, pdb_id: str, pocket_name: str) -> str:
    candidates = [
        os.path.join(pdbbind_root, pdb_id, f"{pdb_id}_protein.pdb"),
        os.path.join(pdbbind_root, pdb_id, pocket_name),
        os.path.join(pdbbind_root, f"{pdb_id}_protein.pdb"),
        os.path.join(pdbbind_root, pocket_name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Reference file not found for {pdb_id} (protein/pocket).")


# -------------------- reference chain extraction --------------------
def extract_best_reference_chain(ref_path: str, requested_chain: str, info_seq1: Optional[str] = None):
    """
    Return (seq1_full, CA_xyz_full, resnums_full, chain_id_selected).
    Strategy:
    - Try the requested chain first.
    - If it fails (no CA), scan all chains and pick the one with best local alignment
      score to info_seq1 (if provided) + tie-break by #CAs.
    - Use is_aa(..., standard=False) to keep modified residues (MSE, etc.).
    """
    parser = MMCIFParser(QUIET=True) if ref_path.lower().endswith((".cif", ".mmcif")) \
        else PDBParser(QUIET=True, PERMISSIVE=True)
    struct = parser.get_structure(os.path.basename(ref_path), ref_path)
    model = list(struct.get_models())[0]

    def chain_to_seq_ca(chain):
        seq1, xyz, resnums = [], [], []
        for res in chain:
            if not is_aa(res, standard=False):
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
            resnums.append(res.id[1])  # residue number (resseq)
        return "".join(seq1), np.asarray(xyz, float), np.asarray(resnums, int)

    # Try requested chain
    preferred = None
    for ch in model.get_chains():
        if ch.id.strip() == requested_chain.strip():
            preferred = ch
            break

    candidates = []
    if preferred is not None:
        s, x, r = chain_to_seq_ca(preferred)
        if len(s) >= 1:
            candidates.append(("preferred", preferred.id, s, x, r))

    # Scan all chains as fallback
    for ch in model.get_chains():
        s, x, r = chain_to_seq_ca(ch)
        if len(s) >= 1:
            candidates.append(("scan", ch.id, s, x, r))

    if not candidates:
        raise ValueError(f"No protein CA found in reference: {ref_path}")

    # Score: local alignment score vs info_seq1 (if available), tie-break by length
    best = None
    best_score = -1e9
    for tag, cid, s, x, r in candidates:
        score = 0.0
        if info_seq1:
            alns = pairwise2.align.localms(s, info_seq1, 2, -1, -5, -1, one_alignment_only=True)
            if alns:
                score = alns[0].score
        score += 0.001 * len(s)
        if score > best_score:
            best_score = score
            best = (cid, s, x, r)

    sel_chain, sel_seq, sel_xyz, sel_resnums = best
    return sel_seq, sel_xyz, sel_resnums, sel_chain


# -------------------- AF3 model reading --------------------
def list_af3_models(folder: str, pdb_id: str) -> Dict[str, str]:
    out = {}
    if not os.path.isdir(folder):
        return out
    pat = re.compile(r"(?:^|_)model_([0-9]+)\.(?:cif|pdb)$", re.IGNORECASE)
    for nm in os.listdir(folder):
        if f"fold_{pdb_id}" in nm.lower():
            m = pat.search(nm)
            if m:
                out[f"model_{m.group(1)}"] = os.path.join(folder, nm)
    if not out:
        for nm in os.listdir(folder):
            m = pat.search(nm)
            if m:
                out[f"model_{m.group(1)}"] = os.path.join(folder, nm)
    return out

def read_predicted_chains(pred_path: str) -> List[Tuple[str, str, np.ndarray]]:
    parser = MMCIFParser(QUIET=True) if pred_path.lower().endswith((".cif", ".mmcif")) \
        else PDBParser(QUIET=True, PERMISSIVE=True)
    struct = parser.get_structure(os.path.basename(pred_path), pred_path)
    model = list(struct.get_models())[0]
    results = []
    for ch in model.get_chains():
        seq1, xyz = [], []
        for res in ch:
            if not is_aa(res, standard=False):
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
            results.append((ch.id, "".join(seq1), np.asarray(xyz, float)))
    return results


# -------------------- alignment & RMSD --------------------
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


# -------------------- per-target compute --------------------
def compute_one(pdb_id: str, rec: dict, args, rows_models: list, rows_best: list):
    ref_path = find_reference_protein(args.pdbbind_root, pdb_id, rec["pocket"])
    ref_seq_full, ref_xyz_full, ref_resnums, ref_chain_id = extract_best_reference_chain(
        ref_path, rec["chain"], rec.get("seq1", "")
    )

    af3_dir = os.path.join(args.af3_root, pdb_id)
    models = list_af3_models(af3_dir, pdb_id)
    if not models:
        raise FileNotFoundError(f"No AF3 models in {af3_dir}")

    best_row = None
    best_value = None
    metric_name = "rmsd_rigid_A" if args.mode in ("rigid", "both") else "rmsd_scale_A"

    for tag, fpath in sorted(models.items(), key=lambda kv: int(kv[0].split("_")[-1])):
        preds = read_predicted_chains(fpath)
        if not preds:
            rows_models.append(dict(pdb_id=pdb_id, model=tag, file=os.path.basename(fpath),
                                    ref_chain=ref_chain_id, n_ref_window=rec["end"] - rec["start"] + 1,
                                    n_match=0, error="no_pred_chain"))
            continue

        # choose predicted chain that yields max overlap inside [start, end] window
        chosen = None
        chosen_overlap = -1
        chosen_ip = chosen_ir = None

        for ch_id, seq_pred, xyz_pred in preds:
            ip_full, ir_full = align_local_indices(seq_pred, ref_seq_full)
            if len(ip_full) < 3:
                continue
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
                   ref_chain=ref_chain_id, n_ref_window=rec["end"] - rec["start"] + 1,
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
            val = row[metric_name]
            if best_value is None or val < best_value:
                best_value = val
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
    # Defaults for IDE runs
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




