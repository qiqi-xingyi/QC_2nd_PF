# --*-- conding:utf-8 --*--
# @time:9/3/25 03:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:get_seq.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract amino-acid sequences from your XYZ folders and write FASTA + CSV.

Assumptions about your XYZ:
- First line may be an integer (atom/residue count) -> skip it if so.
- Each subsequent line starts with a single-letter amino-acid code (A,C,D,...,Y),
  followed by three floats (x,y,z). Example:
      A 0.0 0.0 0.0
      L 0.57 0.57 -0.57
- All files inside one folder share the same sequence (we read <id>.xyz if present,
  otherwise fallback to <id>_top_1.xyz).

Input  root: data/Quantum_original_data/
Outputs:
  - data/seqs/<id>.fasta
  - data/seqs.csv  (id,sequence)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

AA_20 = set(list("ACDEFGHIKLMNPQRSTVWY") + ["O", "U", "B", "Z", "X"])  # be tolerant

def read_sequence_from_xyz(xyz_path: Path) -> str:
    """Parse sequence from a custom XYZ file: first token per line is the residue code."""
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ not found: {xyz_path}")
    lines = xyz_path.read_text().splitlines()
    seq_letters: List[str] = []

    # skip first line if it looks like an integer count
    start = 1 if (len(lines) > 0 and lines[0].strip().isdigit()) else 0

    for ln in lines[start:]:
        parts = ln.strip().split()
        if len(parts) < 1:
            continue
        aa = parts[0].upper()
        # only accept 1-letter amino-acid codes; warn but keep uncommon letters (X,B,Z,U,O)
        if len(aa) == 1 and aa in AA_20:
            seq_letters.append(aa)
        else:
            # stop at the first non-AA token (robust to trailing blank lines)
            # alternatively, you could 'continue' to be even more tolerant
            break

    if not seq_letters:
        raise ValueError(f"No amino-acid letters parsed in {xyz_path}")
    return "".join(seq_letters)

def choose_xyz_for_id(id_dir: Path) -> Path:
    """Prefer <id>.xyz; if missing, fallback to <id>_top_1.xyz."""
    pid = id_dir.name
    main_xyz = id_dir / f"{pid}.xyz"
    if main_xyz.exists():
        return main_xyz
    top1 = id_dir / f"{pid}_top_1.xyz"
    if top1.exists():
        return top1
    # last resort: any *.xyz
    xs = sorted(id_dir.glob("*.xyz"))
    if xs:
        return xs[0]
    raise FileNotFoundError(f"No XYZ files under {id_dir}")

def write_fasta(seq_id: str, seq: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{seq_id}.fasta"
    with out.open("w") as f:
        f.write(f">{seq_id}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")
    return out

def main():
    root_in = Path("data/Quantum_original_data").resolve()
    out_fasta_dir = Path("data/seqs").resolve()
    out_csv = Path("data/seqs.csv").resolve()

    if not root_in.exists():
        raise SystemExit(f"[ERROR] Input root not found: {root_in}")

    rows: List[Tuple[str, str]] = []
    ids_processed = 0

    for id_dir in sorted(root_in.iterdir()):
        if not id_dir.is_dir():
            continue
        pid = id_dir.name
        try:
            xyz = choose_xyz_for_id(id_dir)
            seq = read_sequence_from_xyz(xyz)
            write_fasta(pid, seq, out_fasta_dir)
            rows.append((pid, seq))
            ids_processed += 1
            print(f"[OK] {pid}: length={len(seq)} -> {out_fasta_dir / (pid + '.fasta')}")
        except Exception as e:
            print(f"[WARN] skip {pid}: {e}")

    if not rows:
        raise SystemExit("[ERROR] No sequences extracted; check your paths and XYZ format.")

    # write CSV (id,sequence)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("id,sequence\n")
        for pid, seq in rows:
            f.write(f"{pid},{seq}\n")

    print(f"[DONE] Extracted {ids_processed} sequences.")
    print(f"[INFO] FASTA dir: {out_fasta_dir}")
    print(f"[INFO] CSV: {out_csv}")

if __name__ == "__main__":
    main()
