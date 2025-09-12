# --*-- conding:utf-8 --*--
# @time:9/11/25 21:38
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:rmsd_quantum_original.py.py

"""
Compute RMSD for pure-quantum candidates (top1 & top1..top5) against PDBbind pockets.

Assumptions
- Predicted xyz are residue-level points: lines like "<AA> x y z".
- Treat them as pseudo Cα; pre-scale so median adjacent distance = 3.8 Å.
- Rigid RMSD (Kabsch) and optional similarity RMSD (uniform scale).
- "Top1" is FIXED as the file whose name contains "_top_1" (not by RMSD).

Inputs
  info.txt:  "<id> <pocket.pdb> Chain <X> Residues <a-b> <SEQ-3letter-...>"
  PDBbind:   data/Pdbbind/<id>/<id>_pocket.pdb  (or directly under root)
  Quantum:   data/Quantum_original_data/<id>/*_top_1.xyz ... *_top_5.xyz

Outputs
  <out_dir>/quantum_top1.csv
  <out_dir>/quantum_top5.csv
"""

import os, re, argparse
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from Bio.PDB import PDBParser, MMCIFParser, is_aa
try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    from Bio.Data.IUPACData import protein_letters_3to1 as _3to1_dict
    def three_to_one(resname: str) -> str:
        """Map three-letter residue name to one-letter code."""
        resname = resname.strip().upper()
        return _3to1_dict.get(resname, "X")

from Bio import pairwise2

AA1 = set(list("ACDEFGHIKLMNPQRSTVWY"))
CA_CA_TARGET = 3.8  # Å

# ---------- info.txt parsing ----------
def parse_info_line(line: str):
    parts = line.strip().split()
    if not parts or len(parts) < 4:
        return None
    pdb_id = parts[0].lower()
    pdb_file = parts[1]
    # chain
    if parts[2].lower() == "chain":
        chain = parts[3]; idx = 4
    else:
        chain = parts[2]; idx = 3
    # range
    if idx < len(parts) and parts[idx].lower() == "residues":
        rng = parts[idx+1];  # "a-b"
    else:
        rng = parts[idx]
    rng = rng.replace(",", "")
    a, b = rng.split("-")
    start, end = int(a), int(b)
    return dict(pdb_id=pdb_id, pdb_file=pdb_file, chain=chain, start=start, end=end)

def load_info(path: str) -> Dict[str, dict]:
    info = {}
    with open(path, "r") as f:
        for ln in f:
            if not ln.strip(): continue
            rec = parse_info_line(ln)
            if rec: info[rec["pdb_id"]] = rec
    return info

# ---------- residue-level xyz ----------
def read_residue_xyz(path: str) -> Tuple[str, np.ndarray]:
    lines = [l.strip() for l in open(path, "r") if l.strip()]
    i = 0
    # optional integer atom-count line
    try:
        int(lines[0].split()[0]); i = 1
    except Exception:
        i = 0
    def looks_row(s):
        p = s.split()
        if len(p) < 4: return False
        if len(p[0]) != 1: return False
        try:
            float(p[1]); float(p[2]); float(p[3])
        except Exception:
            return False
        return True
    if i < len(lines) and not looks_row(lines[i]):
        i += 1
    seq, coords = [], []
    for l in lines[i:]:
        p = l.split()
        if len(p) < 4: continue
        aa = p[0].upper()
        if len(aa) != 1 or aa not in AA1: continue
        x, y, z = map(float, p[1:4])
        seq.append(aa); coords.append([x, y, z])
    if not seq:
        raise ValueError(f"Failed to parse xyz: {path}")
    return "".join(seq), np.asarray(coords, float)

# ---------- pocket Cα fragment ----------
def extract_fragment_ca(pdb_path: str, chain_id: str, start: int, end: int):
    parser = MMCIFParser(QUIET=True) if pdb_path.lower().endswith((".cif",".mmcif")) \
        else PDBParser(QUIET=True, PERMISSIVE=True)
    struct = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    model = list(struct.get_models())[0]
    target = None
    for ch in model.get_chains():
        if ch.id.strip() == chain_id.strip():
            target = ch; break
    if target is None:
        raise ValueError(f"Chain {chain_id} not found in {pdb_path}")
    seq1, xyz = [], []
    for res in target:
        if not is_aa(res, standard=True): continue
        rno = res.id[1]
        if rno < start or rno > end: continue
        if "CA" not in res: continue
        try:
            one = three_to_one(res.get_resname())
        except KeyError:
            continue
        seq1.append(one)
        xyz.append(res["CA"].coord)
    if not xyz:
        raise ValueError(f"No Cα in {pdb_path}:{chain_id} {start}-{end}")
    return "".join(seq1), np.asarray(xyz, float)

# ---------- alignment & RMSD ----------
def align_indices(seq_pred: str, seq_ref: str):
    alns = pairwise2.align.globalms(seq_pred, seq_ref, 1, -1, -5, -1, one_alignment_only=True)
    if not alns: raise RuntimeError("Sequence alignment failed")
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

# ---------- pre-scale to 3.8 Å ----------
def prenorm_scale_to_caca(X: np.ndarray, target: float = CA_CA_TARGET):
    if len(X) < 2: return X.copy(), 1.0
    d = np.linalg.norm(np.diff(X, axis=0), axis=1)
    med = float(np.median(d))
    if med <= 0: return X.copy(), 1.0
    s = target / med
    return X * s, s

# ---------- helpers ----------
def list_top_xyz(qdir: str) -> Dict[str, str]:
    """Return mapping: 'top_1'..'top_5' -> filepath, by fuzzy matching '_top_[1-5]'.
    Works with names like '01_1bai_top_1.xyz' or '1bai_top_1.xyz'."""
    out = {}
    for nm in os.listdir(qdir):
        low = nm.lower()
        if not low.endswith((".xyz", ".txt")):
            continue
        m = re.search(r"_top_([1-5])(?:\D|$)", low)
        if m:
            out[f"top_{m.group(1)}"] = os.path.join(qdir, nm)
    return out

def find_pocket_path(pdbbind_root: str, pdb_id: str, pocket_name: str) -> str:
    cand = os.path.join(pdbbind_root, pdb_id, pocket_name)
    if os.path.exists(cand): return cand
    cand2 = os.path.join(pdbbind_root, pocket_name)
    if os.path.exists(cand2): return cand2
    raise FileNotFoundError(f"Pocket not found: {pdb_id}/{pocket_name}")

# ---------- main per target ----------
def compute_one(pdb_id: str, rec: dict, args, rows_top5: list, rows_top1: list):
    qdir = os.path.join(args.quantum_root, pdb_id)
    if not os.path.isdir(qdir):
        raise FileNotFoundError(f"Quantum folder missing: {qdir}")

    pocket_path = find_pocket_path(args.pdbbind_root, pdb_id, rec["pdb_file"])
    ref_seq, ref_ca = extract_fragment_ca(pocket_path, rec["chain"], rec["start"], rec["end"])

    files = list_top_xyz(qdir)
    if not files:
        raise FileNotFoundError(f"No *_top_[1-5].xyz in {qdir}")

    # —— compute RMSD for all available top_k —— #
    cache_top1_row = None

    for tag in sorted(files.keys(), key=lambda s: int(s.split("_")[-1])):
        fpath = files[tag]
        seq_pred, X_pred = read_residue_xyz(fpath)
        X_pred, s_pre = prenorm_scale_to_caca(X_pred, CA_CA_TARGET)

        ip, ir = align_indices(seq_pred, ref_seq)
        row = dict(pdb_id=pdb_id, tag=tag, xyz=os.path.basename(fpath),
                   n_pred=len(seq_pred), n_ref=len(ref_seq), n_match=len(ip),
                   pre_scale=round(s_pre, 4))

        if len(ip) >= 3:
            P, Q = X_pred[ip], ref_ca[ir]
            if args.mode in ("rigid", "both"):
                row["rmsd_rigid_A"] = round(kabsch_rmsd(P, Q), 3)
            if args.mode in ("scale", "both"):
                r_sim, s = similarity_rmsd(P, Q)
                row["rmsd_scale_A"] = round(r_sim, 3)
                row["similarity_scale"] = round(s, 4)
        else:
            row["error"] = "too_few_matches"

        rows_top5.append(row)

        if tag == "top_1":
            cache_top1_row = row

    # —— fixed top1 from filename —— #
    if cache_top1_row is None:
        raise FileNotFoundError(f"{pdb_id}: missing file with '_top_1'")

    row_top1 = dict(pdb_id=pdb_id, chosen_tag="top_1",
                    xyz=cache_top1_row.get("xyz"),
                    n_pred=cache_top1_row.get("n_pred"),
                    n_ref=cache_top1_row.get("n_ref"),
                    n_match=cache_top1_row.get("n_match"),
                    pre_scale=cache_top1_row.get("pre_scale"))

    for k in ("rmsd_rigid_A", "rmsd_scale_A", "similarity_scale", "error"):
        if k in cache_top1_row:
            row_top1[k] = cache_top1_row[k]

    rows_top1.append(row_top1)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="RMSD for Quantum-original (top1 & top1..top5)")
    ap.add_argument("--info", required=True)
    ap.add_argument("--pdbbind-root", required=True)
    ap.add_argument("--quantum-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["rigid", "scale", "both"], default="rigid",
                    help="Which RMSD to compute")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    info = load_info(args.info)

    rows_top1, rows_top5 = [], []

    for pdb_id in sorted(os.listdir(args.quantum_root)):
        pid = pdb_id.lower()
        if pid not in info:
            continue
        try:
            compute_one(pid, info[pid], args, rows_top5, rows_top1)
            print(f"[OK] {pid}")
        except Exception as e:
            print(f"[ERR] {pid}: {e}")
            rows_top1.append(dict(pdb_id=pid, error=str(e)))

    f1 = os.path.join(args.out_dir, "quantum_top1.csv")
    f5 = os.path.join(args.out_dir, "quantum_top5.csv")
    pd.DataFrame(rows_top1).to_csv(f1, index=False)
    pd.DataFrame(rows_top5).to_csv(f5, index=False)
    print(f"Saved: {f1}")
    print(f"Saved: {f5}")

if __name__ == "__main__":
    main()

