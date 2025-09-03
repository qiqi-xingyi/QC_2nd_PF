# --*-- conding:utf-8 --*--
# @time:9/3/25 15:02
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:filter_seqs_minlen10.py

"""
Filter sequences by minimum length (default: 10 aa).

Input:
  <project_root>/data/seqs/*.fasta

Outputs:
  <project_root>/data/seqs_len10/           # FASTA copies with len >= 10
  <project_root>/data/seqs_len10.csv        # id,sequence for kept set
  <project_root>/data/seqs_excluded_lt10.csv# id,sequence for excluded set
"""

from __future__ import annotations
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "data" / "seqs"
DST_DIR      = PROJECT_ROOT / "data" / "seqs_len10"
CSV_KEEP     = PROJECT_ROOT / "data" / "seqs_len10.csv"
CSV_EXCL     = PROJECT_ROOT / "data" / "seqs_excluded_lt10.csv"
MIN_LEN      = 10

def read_fasta_seq(path: Path) -> str:
    lines = path.read_text().splitlines()
    seq_lines = []
    for ln in lines:
        if not ln:
            continue
        if ln.startswith(">"):
            continue
        seq_lines.append(ln.strip())
    return "".join(seq_lines).replace(" ", "").upper()

def main():
    if not SRC_DIR.exists():
        print(f"[ERROR] Not found: {SRC_DIR}")
        return

    DST_DIR.mkdir(parents=True, exist_ok=True)
    keep_rows = []
    excl_rows = []

    fas = sorted(SRC_DIR.glob("*.fasta"))
    if not fas:
        print(f"[ERROR] No FASTA files in {SRC_DIR}")
        return

    kept, excluded = 0, 0
    for fa in fas:
        sid = fa.stem
        seq = read_fasta_seq(fa)
        L = len(seq)
        if L >= MIN_LEN:
            # copy fasta
            shutil.copy2(fa, DST_DIR / fa.name)
            keep_rows.append((sid, seq))
            kept += 1
            print(f"[KEEP] {sid}: len={L}")
        else:
            excl_rows.append((sid, seq))
            excluded += 1
            print(f"[SKIP] {sid}: len={L} (<{MIN_LEN})")

    # write CSVs
    CSV_KEEP.parent.mkdir(parents=True, exist_ok=True)
    with CSV_KEEP.open("w") as f:
        f.write("id,sequence\n")
        for sid, seq in keep_rows:
            f.write(f"{sid},{seq}\n")

    with CSV_EXCL.open("w") as f:
        f.write("id,sequence\n")
        for sid, seq in excl_rows:
            f.write(f"{sid},{seq}\n")

    print("\n[DONE]")
    print(f"  kept (>= {MIN_LEN}): {kept}  -> {CSV_KEEP}")
    print(f"  excluded (< {MIN_LEN}): {excluded}  -> {CSV_EXCL}")
    print(f"  FASTA copied to: {DST_DIR}")

if __name__ == "__main__":
    main()
