# --*-- conding:utf-8 --*--
# @time:9/9/25 21:29
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:make_multifasta.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a multi-FASTA from:
  - data/seqs_len10.csv (columns: id,sequence)
  - data/seqs_len10/*.fasta (fallback for ids not in CSV)
Output: data/seqs_len10_all.fasta
Rules:
  - CSV has priority over per-file FASTA
  - Uppercase sequences; keep only ACDEFGHIKLMNPQRSTVWY
"""

from pathlib import Path
import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(s: str) -> str:
    s = (s or "").strip().upper().replace(" ", "")
    return "".join([c for c in s if c in AA])

def read_csv(csv_path: Path) -> dict:
    records = {}
    if not csv_path.exists():
        return records
    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        sid = str(r["id"]).strip()
        seq = clean_seq(str(r["sequence"]))
        if sid and seq:
            records[sid] = seq
    return records

def read_single_fastas(fasta_dir: Path) -> dict:
    records = {}
    if not fasta_dir.exists():
        return records
    for fa in sorted(fasta_dir.glob("*.fasta")):
        text = fa.read_text().strip()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines or not lines[0].startswith(">"):
            continue
        sid = lines[0][1:].strip().split()[0]
        seq = clean_seq("".join(lines[1:]))
        if sid and seq:
            records[sid] = seq
    return records

def write_multifasta(records: dict, out_fa: Path) -> int:
    out_fa.parent.mkdir(parents=True, exist_ok=True)
    with out_fa.open("w") as f:
        for sid, seq in records.items():
            f.write(f">{sid}\n{seq}\n")
    return len(records)

def main():
    proj = Path(__file__).resolve().parents[1]
    csv_path = proj / "data" / "seqs_len10.csv"
    fasta_dir = proj / "data" / "seqs_len10"
    out_fa = proj / "data" / "seqs_len10_all.fasta"

    records = read_csv(csv_path)
    fallback = read_single_fastas(fasta_dir)
    # CSV has priority; add only ids not present
    for sid, seq in fallback.items():
        if sid not in records:
            records[sid] = seq

    n = write_multifasta(records, out_fa)
    print(f"[make_multifasta] wrote {n} sequences to {out_fa}")

if __name__ == "__main__":
    main()
