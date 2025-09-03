# --*-- conding:utf-8 --*--
# @time:9/3/25 03:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:get_seq.py

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional

# ====== CONFIG (edit here if needed) =========================================
# go up one level from tools/ to project root
ROOT_IN = Path(__file__).resolve().parent.parent / "data" / "Quantum_original_data"
OUT_FASTA_DIR = Path(__file__).resolve().parent.parent / "data" / "seqs"
OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "seqs.csv"
CHECK_CONSISTENCY = True                       # warn if top_k sequences differ
# ============================================================================

# be tolerant: include rare codes often seen in predicted data
AA_CODES = set(list("ACDEFGHIKLMNPQRSTVWY") + ["O", "U", "B", "Z", "X"])

def read_sequence_from_xyz(xyz_path: Path) -> str:
    """
    Parse sequence from a custom XYZ file:
      - First line may be an integer count -> skip it.
      - Each subsequent line starts with a single-letter amino-acid code, then x y z.
    """
    lines = xyz_path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty file: {xyz_path}")

    start = 1 if lines[0].strip().isdigit() else 0
    seq_letters: List[str] = []

    for ln in lines[start:]:
        parts = ln.strip().split()
        if not parts:
            continue
        aa = parts[0].upper()
        if len(aa) == 1 and aa in AA_CODES:
            seq_letters.append(aa)
        else:
            # stop at the first non-AA token; avoids reading coordinate-only lines
            break

    if not seq_letters:
        raise ValueError(f"No amino-acid letters parsed in {xyz_path}")
    return "".join(seq_letters)

def choose_xyz_for_id(id_dir: Path) -> Optional[Path]:
    """
    Prefer <id>.xyz; fallback to <id>_top_1.xyz; otherwise any *.xyz.
    Return None if there is no xyz file at all.
    """
    pid = id_dir.name
    main_xyz = id_dir / f"{pid}.xyz"
    if main_xyz.exists():
        return main_xyz
    top1 = id_dir / f"{pid}_top_1.xyz"
    if top1.exists():
        return top1
    xs = sorted(id_dir.glob("*.xyz"))
    return xs[0] if xs else None

def write_fasta(seq_id: str, seq: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{seq_id}.fasta"
    with out.open("w") as f:
        f.write(f">{seq_id}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")
    return out

def check_all_xyz_sequences(id_dir: Path, base_seq: str) -> None:
    """
    Optional: check whether all *.xyz under this id share the same sequence.
    Only warns; does not fail.
    """
    diffs = []
    for xyz in sorted(id_dir.glob("*.xyz")):
        try:
            s = read_sequence_from_xyz(xyz)
            if s != base_seq:
                diffs.append(xyz.name)
        except Exception:
            continue
    if diffs:
        print(f"[WARN] {id_dir.name}: sequence differs in files: {', '.join(diffs)}")

def main():
    if not ROOT_IN.exists():
        print(f"[ERROR] Input root not found: {ROOT_IN.resolve()}")
        return

    OUT_FASTA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, str]] = []
    processed = 0
    skipped = 0

    for id_dir in sorted(ROOT_IN.iterdir()):
        if not id_dir.is_dir():
            continue
        pid = id_dir.name
        try:
            xyz = choose_xyz_for_id(id_dir)
            if xyz is None:
                print(f"[WARN] {pid}: no .xyz file found, skip.")
                skipped += 1
                continue
            seq = read_sequence_from_xyz(xyz)
            write_fasta(pid, seq, OUT_FASTA_DIR)
            rows.append((pid, seq))
            processed += 1
            print(f"[OK] {pid}: length={len(seq)} → {OUT_FASTA_DIR / (pid + '.fasta')}")
            if CHECK_CONSISTENCY:
                check_all_xyz_sequences(id_dir, seq)
        except Exception as e:
            print(f"[WARN] skip {pid}: {e}")
            skipped += 1

    if not rows:
        print("[ERROR] No sequences extracted; please check input format/paths.")
        return

    # write CSV
    with OUT_CSV.open("w") as f:
        f.write("id,sequence\n")
        for pid, seq in rows:
            f.write(f"{pid},{seq}\n")

    print("\n[DONE]")
    print(f"  Sequences extracted: {processed}")
    print(f"  Folders skipped:     {skipped}")
    print(f"  FASTA dir:           {OUT_FASTA_DIR.resolve()}")
    print(f"  CSV:                 {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()


