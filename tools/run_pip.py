# --*-- conding:utf-8 --*--
# @time:9/3/25 03:38
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:run_pip.py

"""
End-to-end runner:
- Read sequences from data/seqs
- Generate AI priors via NN_layer
- Rescore existing VQE candidates via Mid_layer

Directory assumptions (all paths resolved relative to project root):
  data/
    seqs/                 # produced by extract_sequences.py (FASTA files)
    seqs.csv              # optional summary (id,sequence)
  runs/
    exp001/
      priors/             # <id>.prior.npz will be placed/updated here
  Result/
    process_data/
      best_group/
        <id>/
          <id>.xyz
          <id>_top_{1..5}.xyz
          top_5_energies_<id>.txt   # optional

This script is safe to run multiple times.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import csv
import sys

# --- Resolve project root no matter where this script is launched from ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Config (edit if needed) ----------------------------------------------
SEQ_DIR        = PROJECT_ROOT / "data" / "seqs"                       # where <id>.fasta live
SEQ_CSV        = PROJECT_ROOT / "data" / "seqs.csv"                   # optional
PRIORS_RUNROOT = PROJECT_ROOT / "runs" / "exp001"                     # priors/<id>.prior.npz under here
RESULTS_ROOT   = PROJECT_ROOT / "Result" / "process_data" / "best_group"
OUT_CSV        = PROJECT_ROOT / "rescored_results.csv"
LAMBDA         = 0.4                                                  # weight for E_SS
USE_PER_RESIDUE_MU = True                                             # use (phi_mu, psi_mu)
SKIP_PRIOR_IF_EXISTS = True                                           # do not recompute if file exists
WRITE_SS_BEST  = True                                                 # copy <id>_ss_best.xyz
# -------------------------------------------------------------------------

# --- Optional NN layer (sequence -> prior) ---
try:
    from NN_layer import run_pipeline  # (seq_dir: Path, out_root: Path, run_id: str) -> Path
except Exception:
    run_pipeline = None

# --- Mid layer (rescoring) ---
try:
    from Mid_layer import Rescorer, RescoreConfig
except Exception as e:
    print(f"[ERROR] Mid_layer not importable: {e}", file=sys.stderr)
    raise

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def count_fasta_files(d: Path) -> int:
    return len(list(d.glob("*.fasta"))) if d.exists() else 0

def generate_priors_from_seqs(seq_dir: Path, seq_csv: Optional[Path], priors_root: Path) -> Path:
    """
    Trigger NN_layer to generate priors. If SKIP_PRIOR_IF_EXISTS is True and all priors already
    exist (based on seq_dir filenames), the call will be skipped.
    """
    if run_pipeline is None:
        print("[WARN] NN_layer.run_pipeline not available. Skipping prior generation.")
        return priors_root / "priors"

    ensure_dir(priors_root)
    priors_dir = priors_root / "priors"

    # Check whether priors already exist (by FASTA basenames)
    if SKIP_PRIOR_IF_EXISTS and seq_dir.exists():
        missing = []
        for fa in seq_dir.glob("*.fasta"):
            pid = fa.stem
            if not (priors_dir / f"{pid}.prior.npz").exists():
                missing.append(pid)
        if not missing and priors_dir.exists():
            print(f"[INFO] All priors exist under {priors_dir}. Skip NN inference.")
            return priors_dir
        else:
            if missing:
                print(f"[INFO] Priors missing for {len(missing)} ids; will run NN_layer.")

    # If no FASTA but we have CSV, bail out with hint
    if not seq_dir.exists() or count_fasta_files(seq_dir) == 0:
        if seq_csv and seq_csv.exists():
            print(f"[ERROR] No FASTA in {seq_dir}. Please run your sequence extraction (or convert {seq_csv} to FASTA).")
        else:
            print(f"[ERROR] No FASTA in {seq_dir}.")
        sys.exit(1)

    print(f"[INFO] Running NN_layer.run_pipeline on {seq_dir} ...")
    out_dir = run_pipeline(seq_dir, priors_root, run_id="exp001")
    print(f"[OK] Priors generated under: {out_dir}")
    return out_dir

def rescore_all(results_root: Path, priors_dir: Path, out_csv: Path, lam: float, use_mu: bool, write_ss_best: bool) -> None:
    """
    Iterate over <id> folders under results_root, use priors/<id>.prior.npz to rescore candidates,
    and write a CSV with detailed energy terms. Optionally copy ss-best XYZ.
    """
    if not results_root.exists():
        print(f"[ERROR] Results root not found: {results_root}")
        sys.exit(1)
    if not priors_dir.exists():
        print(f"[ERROR] Priors dir not found: {priors_dir}")
        sys.exit(1)

    ensure_dir(out_csv.parent)
    rows = []
    processed = 0
    skipped = 0

    for id_dir in sorted(results_root.iterdir()):
        if not id_dir.is_dir():
            continue
        pid = id_dir.name
        prior_path = priors_dir / f"{pid}.prior.npz"
        if not prior_path.exists():
            print(f"[WARN] Missing prior: {prior_path}. Skip {pid}.")
            skipped += 1
            continue

        resc = Rescorer(id_dir, prior_path, cfg=RescoreConfig(lam=lam, use_per_residue_mu=use_mu))
        ranked = resc.rescore()

        ss_best_path = None
        if write_ss_best and ranked:
            ss_best_path = resc.save_ss_best(ranked)

        for c in ranked:
            rows.append({
                "protein_id": pid,
                "candidate": c.tag,
                "xyz_path": str(c.xyz),
                "prior_path": str(prior_path),
                "lambda": resc.cfg.lam,
                "E_Q": "" if c.E_Q is None else f"{c.E_Q:.8f}",
                "E_torsion": f"{c.terms['E_torsion']:.8f}",
                "E_alphaHB": f"{c.terms['E_alphaHB']:.8f}",
                "E_betaHB":  f"{c.terms['E_betaHB']:.8f}",
                "E_SS":      f"{float(c.E_SS):.8f}",
                "E_total":   f"{float(c.E_total):.8f}",
                "alpha_hb_sum": f"{c.terms['alpha_hb_sum']:.8f}",
                "beta_hb_sum":  f"{c.terms['beta_hb_sum']:.8f}",
                "phi_rmse":  f"{c.terms['phi_rmse']:.6f}",
                "psi_rmse":  f"{c.terms['psi_rmse']:.6f}",
                "P_alpha_mean": f"{c.terms['P_alpha_mean']:.6f}",
                "P_beta_mean":  f"{c.terms['P_beta_mean']:.6f}",
                "P_coil_mean":  f"{c.terms['P_coil_mean']:.6f}",
                "picked_by_AI": "1" if c is ranked[0] else "0",
                "picked_by_VQE": "1" if c.tag == "best" else "0",
                "ss_best_xyz": "" if (ss_best_path is None or c is not ranked[0]) else str(ss_best_path),
            })
        processed += 1
        print(f"[OK] {pid}: ranked {len(ranked)} candidates.{' ss_best saved.' if ss_best_path else ''}")

    if not rows:
        print("[WARN] Nothing to write (no candidates or no priors).")
        return

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n[DONE]")
    print(f"  Processed ids: {processed}")
    print(f"  Skipped ids:   {skipped}")
    print(f"  Output CSV:    {out_csv}")

def main():
    print("[STEP 1] Generate/ensure priors ...")
    priors_dir = generate_priors_from_seqs(SEQ_DIR, SEQ_CSV, PRIORS_RUNROOT)

    print("[STEP 2] Rescore & re-rank VQE candidates ...")
    rescore_all(
        results_root=RESULTS_ROOT,
        priors_dir=priors_dir,
        out_csv=OUT_CSV,
        lam=LAMBDA,
        use_mu=USE_PER_RESIDUE_MU,
        write_ss_best=WRITE_SS_BEST,
    )

if __name__ == "__main__":
    main()
