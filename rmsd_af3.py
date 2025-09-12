# --*-- conding:utf-8 --*--
# @time:9/12/25 00:11
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:rmsd_af3.py


"""
AF3 short-fragment RMSD vs dataset reference window.

Key design:
- Prefer defining the reference window by sequence alignment to the info.txt sequence,
  to avoid reliance on residue numbering (which may be renumbered in PDBbind files).
- Fall back to numeric window [start,end] only if info sequence is unavailable.
- Relaxed CA-based residue collection on both reference and predicted sides.
- Normalize common modified/alias residue names (MSE->MET, HSD->HIS, SEP->SER, ...).
- If predicted sequence is mostly 'X', use a geometric sliding-window fallback within the window.

Outputs:
  <out_dir>/af3_models.csv   # per (pdb_id, model_k)
  <out_dir>/af3_best.csv     # best per pdb_id by chosen metric
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

from Bio.PDB import PDBParser, MMCIFParser

# robust three_to_one across Biopython versions
try:
    from Bio.PDB.Polypeptide import three_to_one
except Exception:
    from Bio.Data.IUPACData import protein_letters_3to1 as _3to1
    def three_to_one(resname: str) -> str:
        return _3to1.get(resname.strip().upper(), "X")

from Bio import pairwise2
from Bio.Data.IUPACData import protein_letters_3to1 as MAP_3TO1

# -------------------- Config --------------------
RESNAME_NORMALIZE = {
    "MSE": "MET", "SEC": "CYS", "PYL": "LYS",
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "SEP": "SER", "TPO": "THR", "PTR": "TYR",
    "CSO": "CYS", "CME": "CYS", "MLY": "LYS",
    "GLX": "GLU", "ASX": "ASP",
}
PRED_SEQ_UNKNOWN_FRAC = 0.80  # >=80% 'X' in predicted seq -> use geometric fallback
MIN_MATCH = 3

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

# -------------------- residue helpers --------------------
def normalize_resname(resname: str) -> str:
    rn = resname.strip().upper()
    return RESNAME_NORMALIZE.get(rn, rn)

def chain_to_seq_ca_relaxed(chain) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Build (seq1, CA_xyz, resnums) from a biopython Chain:
    - keep any residue that has a CA atom (do not rely on is_aa),
    - normalize residue names, map to one-letter if possible; keep 'X' otherwise.
    """
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
        resnums.append(res.id[1])  # resseq (ignore insertion code)
    return "".join(seq1), np.asarray(xyz, float), np.asarray(resnums, int)

# -------------------- reference chain extraction --------------------
def extract_best_reference_chain(ref_path: str, requested_chain: str, info_seq1: Optional[str] = None,
                                 debug: bool = False):
    parser = MMCIFParser(QUIET=True) if ref_path.lower().endswith((".cif", ".mmcif")) \
        else PDBParser(QUIET=True, PERMISSIVE=True)
    struct = parser.get_structure(os.path.basename(ref_path), ref_path)
    model = list(struct.get_models())[0]

    candidates = []

    # try requested chain first
    preferred = None
    for ch in model.get_chains():
        if ch.id.strip() == requested_chain.strip():
            preferred = ch
            break
    if preferred is not None:
        s, x, r = chain_to_seq_ca_relaxed(preferred)
        if len(s) >= 1:
            candidates.append(("preferred", preferred.id, s, x, r))

    # scan all chains
    for ch in model.get_chains():
        s, x, r = chain_to_seq_ca_relaxed(ch)
        if len(s) >= 1:
            candidates.append(("scan", ch.id, s, x, r))

    if debug:
        print(f"[DEBUG] Reference {os.path.basename(ref_path)} chains (first 10):")
        for tag, cid, s, x, r in candidates[:10]:
            print(f"  - chain {cid:>2} ({tag}), CA={len(s)} seqlen, seq_head={s[:50]}")

    if not candidates:
        raise ValueError(f"No protein CA found in reference: {ref_path}")

    # choose best by local alignment vs info_seq1 (if provided), tie-break by length
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
    if debug:
        print(f"[DEBUG] Selected reference chain: {sel_chain}, CA={len(sel_seq)}")
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

def read_predicted_chains_relaxed(pred_path: str, debug: bool = False) -> List[Tuple[str, str, np.ndarray]]:
    parser = MMCIFParser(QUIET=True) if pred_path.lower().endswith((".cif", ".mmcif")) \
        else PDBParser(QUIET=True, PERMISSIVE=True)
    struct = parser.get_structure(os.path.basename(pred_path), pred_path)
    model = list(struct.get_models())[0]
    results = []
    for ch in model.get_chains():
        seq1, xyz = [], []
        for res in ch:
            if "CA" not in res:
                continue
            rn = normalize_resname(res.get_resname())
            try:
                one = three_to_one(rn)
            except Exception:
                one = "X"
            seq1.append(one)
            xyz.append(res["CA"].coord)
        if seq1 and xyz:
            results.append((ch.id, "".join(seq1), np.asarray(xyz, float)))
    if debug:
        print(f"[DEBUG] Pred {os.path.basename(pred_path)} chains:")
        for cid, s, x in results:
            print(f"  - chain {cid:>2}, CA={len(s)} seqlen, seq_head={s[:50]}")
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

# -------------------- window definition (sequence-first) --------------------
def reference_window_mask(ref_seq_full: str,
                          ref_resnums: np.ndarray,
                          info_seq1: Optional[str],
                          start: int,
                          end: int,
                          debug: bool = False) -> np.ndarray:
    """
    Build a boolean mask over reference indices defining the evaluation window.
    Priority:
      1) If info_seq1 is provided (length>=3): local alignment ref_seq_full vs info_seq1,
         mask = positions where both are aligned (non-gaps).
      2) Else: numeric window mask by ref_resnums in [start,end].
    """
    if info_seq1 and len(info_seq1) >= 3:
        ip, ir = align_local_indices(ref_seq_full, info_seq1)
        mask = np.zeros(len(ref_seq_full), dtype=bool)
        mask[ir] = True
        if debug:
            print(f"[DEBUG] Window by sequence: matched {mask.sum()} residues")
        if mask.sum() >= MIN_MATCH:
            return mask
        # fallback to numeric if too few
        if debug:
            print("[DEBUG] Sequence window too small; falling back to numeric range.")
    # numeric range
    mask = (ref_resnums >= start) & (ref_resnums <= end)
    if debug:
        print(f"[DEBUG] Window by numeric: matched {mask.sum()} residues in [{start},{end}]")
    return mask

# -------------------- geometric sliding-window fallback --------------------
def geometric_window_best(P_pred: np.ndarray, Q_ref_window: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
    """
    Slide a window of length len(P_pred) over Q_ref_window, compute rigid RMSD, pick best.
    Return (P, Q, best_rmsd, ref_start_index) or None.
    """
    Lp = len(P_pred)
    if len(Q_ref_window) < MIN_MATCH or Lp < MIN_MATCH:
        return None
    if Lp > len(Q_ref_window):
        return None
    best_rmsd = None
    best_i = None
    best_Qseg = None
    for i in range(0, len(Q_ref_window) - Lp + 1):
        Qseg = Q_ref_window[i:i+Lp]
        r = kabsch_rmsd(P_pred, Qseg)
        if best_rmsd is None or r < best_rmsd:
            best_rmsd = r
            best_i = i
            best_Qseg = Qseg
    if best_Qseg is None:
        return None
    return P_pred, best_Qseg, float(best_rmsd), best_i

# -------------------- per-target compute --------------------
def compute_one(pdb_id: str, rec: dict, args, rows_models: list, rows_best: list):
    ref_path = find_reference_protein(args.pdbbind_root, pdb_id, rec["pocket"])
    ref_seq_full, ref_xyz_full, ref_resnums, ref_chain_id = extract_best_reference_chain(
        ref_path, rec["chain"], rec.get("seq1", ""), debug=args.debug
    )

    # define reference window (sequence-first)
    mask_win = reference_window_mask(ref_seq_full, ref_resnums, rec.get("seq1", ""), rec["start"], rec["end"], debug=args.debug)
    if mask_win.sum() < MIN_MATCH:
        raise ValueError("Reference window too small after sequence+numeric attempts.")
    Q_window = ref_xyz_full[mask_win]
    ref_idx_window = np.where(mask_win)[0]  # indices in ref_seq_full

    af3_dir = os.path.join(args.af3_root, pdb_id)
    models = list_af3_models(af3_dir, pdb_id)
    if not models:
        raise FileNotFoundError(f"No AF3 models in {af3_dir}")

    best_row = None
    best_value = None
    metric_name = "rmsd_rigid_A" if args.mode in ("rigid", "both") else "rmsd_scale_A"

    for tag, fpath in sorted(models.items(), key=lambda kv: int(kv[0].split("_")[-1])):
        preds = read_predicted_chains_relaxed(fpath, debug=args.debug)
        if not preds:
            rows_models.append(dict(pdb_id=pdb_id, model=tag, file=os.path.basename(fpath),
                                    ref_chain=ref_chain_id, n_ref_window=int(mask_win.sum()),
                                    n_match=0, error="no_pred_chain"))
            continue

        chosen_row = None
        best_metric_val = None
        best_overlap = -1

        for ch_id, seq_pred, xyz_pred in preds:
            Lp = len(seq_pred)
            x_frac = (seq_pred.count("X") / max(1, Lp))

            metric_val = None
            n_match = 0
            rmsd_rigid = None
            rmsd_scale = None
            sim_scale = None

            # sequence-based mapping, but restrict to reference window indices
            used_fallback = False
            if x_frac < PRED_SEQ_UNKNOWN_FRAC:
                ip_full, ir_full = align_local_indices(seq_pred, ref_seq_full)
                if len(ip_full) >= MIN_MATCH:
                    # keep only matches whose ref index is inside window
                    in_win = np.isin(ir_full, ref_idx_window)
                    if in_win.sum() >= MIN_MATCH:
                        ip = ip_full[in_win]
                        ir = ir_full[in_win]
                        P = xyz_pred[ip]
                        Q = ref_xyz_full[ir]
                        n_match = len(ip)
                        if args.mode in ("rigid", "both"):
                            rmsd_rigid = round(kabsch_rmsd(P, Q), 3)
                        if args.mode in ("scale", "both"):
                            r2, s = similarity_rmsd(P, Q)
                            rmsd_scale = round(r2, 3); sim_scale = round(s, 4)
                        metric_val = rmsd_rigid if args.mode in ("rigid", "both") else rmsd_scale
                    else:
                        used_fallback = True
                else:
                    used_fallback = True
            else:
                used_fallback = True

            # geometric fallback: slide over Q_window only
            if used_fallback:
                gw = geometric_window_best(xyz_pred, Q_window)
                if gw is not None:
                    P, Q, rbest, _ = gw
                    n_match = len(P)
                    rmsd_rigid = round(rbest, 3)
                    if args.mode in ("scale", "both"):
                        r2, s = similarity_rmsd(P, Q)
                        rmsd_scale = round(r2, 3); sim_scale = round(s, 4)
                    metric_val = rmsd_rigid if args.mode in ("rigid", "both") else rmsd_scale

            if metric_val is not None:
                if (best_metric_val is None) or (metric_val < best_metric_val) or \
                   (metric_val == best_metric_val and n_match > best_overlap):
                    best_metric_val = metric_val
                    best_overlap = n_match
                    row = dict(pdb_id=pdb_id, model=tag, file=os.path.basename(fpath),
                               ref_chain=ref_chain_id, n_ref_window=int(mask_win.sum()),
                               n_match=n_match, pred_chain=ch_id)
                    if args.mode in ("rigid", "both"):
                        row["rmsd_rigid_A"] = rmsd_rigid
                    if args.mode in ("scale", "both"):
                        row["rmsd_scale_A"] = rmsd_scale
                        row["similarity_scale"] = sim_scale
                    chosen_row = row

        if chosen_row is None:
            rows_models.append(dict(pdb_id=pdb_id, model=tag, file=os.path.basename(fpath),
                                    ref_chain=ref_chain_id, n_ref_window=int(mask_win.sum()),
                                    n_match=0, error="no_valid_alignment"))
        else:
            rows_models.append(chosen_row)
            val = chosen_row["rmsd_rigid_A"] if args.mode in ("rigid", "both") else chosen_row["rmsd_scale_A"]
            if (best_value is None) or (val < best_value):
                best_value = val
                best_row = chosen_row

    rows_best.append(dict(pdb_id=pdb_id, **(best_row if best_row else {"error": "no_valid_model"})))

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="AF3 short-fragment RMSD vs dataset window (sequence-window first)")
    ap.add_argument("--info", required=True)
    ap.add_argument("--pdbbind-root", required=True)
    ap.add_argument("--af3-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["rigid", "scale", "both"], default="rigid")
    ap.add_argument("--debug", action="store_true", help="Print chain summaries and alignment heads")
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
            "--debug",
        ]
    main()