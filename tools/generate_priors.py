# --*-- conding:utf-8 --*--
# @time:9/3/25 11:56
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:generate_priors.py


"""
Generate secondary-structure priors from sequences using NN_layer.

- Reads FASTA files from:   <project_root>/data/seqs/*.fasta
- Writes priors into:       <project_root>/runs/exp001/priors/<id>.prior.npz

Run this file directly in your IDE. No CLI args required.
"""

from __future__ import annotations
from pathlib import Path
import sys

# ---------- Project paths (tools/ and data/ are siblings) ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = PROJECT_ROOT / "data" / "seqs_len10"        # input FASTA directory
PRIORS_ROOT  = PROJECT_ROOT / "runs" / "exp001"       # priors will be placed under this run
RUN_ID       = "exp001"                               # subfolder name under runs/
SKIP_IF_EXISTS = True                                  # do not recompute if <id>.prior.npz exists

# ---------- Import your NN layer pipeline ----------
try:
    # Expected signature: run_pipeline(seq_dir: Path, out_root: Path, run_id: str) -> Path
    from NN_layer import run_pipeline
except Exception as e:
    print(f"[ERROR] Cannot import NN_layer.run_pipeline: {e}", file=sys.stderr)
    sys.exit(1)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def main():
    # Sanity checks
    if not SEQ_DIR.exists():
        print(f"[ERROR] FASTA directory not found: {SEQ_DIR}", file=sys.stderr)
        sys.exit(1)
    fasta_files = sorted(SEQ_DIR.glob("*.fasta"))
    if not fasta_files:
        print(f"[ERROR] No FASTA files under {SEQ_DIR}.", file=sys.stderr)
        sys.exit(1)

    # Prepare output layout
    ensure_dir(PRIORS_ROOT)
    priors_dir = PRIORS_ROOT / "priors"
    ensure_dir(priors_dir)

    # Optionally skip IDs that already have priors
    if SKIP_IF_EXISTS:
        missing = [fa.stem for fa in fasta_files if not (priors_dir / f"{fa.stem}.prior.npz").exists()]
        if not missing:
            print(f"[INFO] All priors already exist under {priors_dir}. Nothing to do.")
            print(f"[DONE] priors_dir = {priors_dir}")
            return
        else:
            print(f"[INFO] Priors missing for {len(missing)} IDs (e.g., {missing[:5]}{'...' if len(missing)>5 else ''}).")

    # Run NN pipeline (this will generate/overwrite priors as needed)
    print(f"[STEP] Running NN_layer.run_pipeline on {SEQ_DIR} ...")
    out_dir = run_pipeline(SEQ_DIR, PRIORS_ROOT, run_id=RUN_ID)
    print(f"[DONE] Priors generated under: {out_dir}")

if __name__ == "__main__":
    main()
