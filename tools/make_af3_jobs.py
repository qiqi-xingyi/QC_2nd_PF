# --*-- conding:utf-8 --*--
# @time:9/9/25 21:57
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:make_af3_jobs.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate AlphaFold Server (AF3) job JSON from local sequence sources.

Inputs (priority: CSV > single FASTA files):
  - data/seqs_len10.csv      (columns: id,sequence)
  - data/seqs_len10/*.fasta  (fallback entries not in CSV)

Output:
  - results/af3_jobs/af3_jobs.json
    or chunked: results/af3_jobs/af3_jobs_part_1.json, ...

Notes:
  - Only 20 standard amino acids are kept (ACDEFGHIKLMNPQRSTVWY).
  - Each sequence becomes a separate job (one protein chain).
  - modelSeeds is an empty list as recommended by AlphaFold Server.
  - You can toggle template usage and set maxTemplateDate if needed.

CLI examples:
  python tools/make_af3_jobs.py
  python tools/make_af3_jobs.py --no-templates
  python tools/make_af3_jobs.py --max-template-date 2018-01-20
  python tools/make_af3_jobs.py --chunk-size 30
"""

from pathlib import Path
import json
import csv
import argparse
from typing import Dict, List, Tuple

AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(s: str) -> str:
    """Uppercase and keep only 20 canonical amino acids."""
    s = (s or "").strip().upper().replace(" ", "")
    return "".join([c for c in s if c in AA])

def read_csv(csv_path: Path) -> Dict[str, str]:
    """Read id,sequence from CSV (utf-8-sig to tolerate BOM)."""
    records: Dict[str, str] = {}
    if not csv_path.exists():
        return records
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = {h.lower(): h for h in (reader.fieldnames or [])}
        id_key = headers.get("id")
        seq_key = headers.get("sequence")
        if not id_key or not seq_key:
            raise RuntimeError(f"CSV must contain 'id,sequence'. Found: {reader.fieldnames}")
        for row in reader:
            sid = str(row[id_key]).strip()
            seq = clean_seq(str(row[seq_key]))
            if sid and seq:
                records[sid] = seq
    return records

def read_single_fastas(fasta_dir: Path) -> Dict[str, str]:
    """Read single-entry FASTA files as fallback."""
    records: Dict[str, str] = {}
    if not fasta_dir.exists():
        return records
    for fa in sorted(fasta_dir.glob("*.fasta")):
        text = fa.read_text(encoding="utf-8", errors="ignore").strip()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines or not lines[0].startswith(">"):
            continue
        sid = lines[0][1:].strip().split()[0]
        seq = clean_seq("".join(lines[1:]))
        if sid and seq:
            records[sid] = seq
    return records

def make_job_dict(name: str,
                  sequence: str,
                  use_templates: bool = True,
                  max_template_date: str = None) -> dict:
    """
    Build a single job dictionary for AlphaFold Server.

    Schema (minimal):
    {
      "name": "<job_name>",
      "modelSeeds": [],
      "sequences": [
        { "proteinChain": { "sequence": "...", "count": 1, ... } }
      ],
      "dialect": "alphafoldserver",
      "version": 1
    }
    """
    protein_chain = {
        "sequence": sequence,
        "count": 1
    }
    # Optional template controls (version 1 fields)
    if use_templates is False:
        protein_chain["useStructureTemplate"] = False
    if isinstance(max_template_date, str) and max_template_date:
        protein_chain["maxTemplateDate"] = max_template_date

    job = {
        "name": name,
        "modelSeeds": [],
        "sequences": [
            {"proteinChain": protein_chain}
        ],
        "dialect": "alphafoldserver",
        "version": 1
    }
    return job

def chunk_list(items: List, n: int) -> List[List]:
    """Split list into chunks of size n (last chunk may be smaller)."""
    if n <= 0:
        return [items]
    return [items[i:i+n] for i in range(0, len(items), n)]

def build_jobs(records: Dict[str, str],
               use_templates: bool,
               max_template_date: str) -> List[dict]:
    """Convert id->sequence dict to AF3 job dicts."""
    jobs: List[dict] = []
    for sid, seq in records.items():
        # Job name can be the sequence id
        job = make_job_dict(
            name=sid,
            sequence=seq,
            use_templates=use_templates,
            max_template_date=max_template_date
        )
        jobs.append(job)
    return jobs

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate AlphaFold Server job JSON.")
    p.add_argument("--csv", type=str, default="data/seqs_len10.csv",
                   help="CSV path (id,sequence). Default: data/seqs_len10.csv")
    p.add_argument("--fasta-dir", type=str, default="data/seqs_len10",
                   help="Directory with single-entry FASTA files. Default: data/seqs_len10")
    p.add_argument("--out-dir", type=str, default="results/af3_jobs",
                   help="Output directory for JSON files. Default: results/af3_jobs")
    p.add_argument("--outfile", type=str, default="af3_jobs.json",
                   help="Output JSON filename (or prefix when chunking). Default: af3_jobs.json")
    p.add_argument("--chunk-size", type=int, default=0,
                   help="If >0, split jobs into multiple JSON files with this many jobs each.")
    p.add_argument("--no-templates", action="store_true",
                   help="Disable structure templates (sets useStructureTemplate=false).")
    p.add_argument("--max-template-date", type=str, default=None,
                   help="Set maxTemplateDate (YYYY-MM-DD), optional. Example: 2018-01-20")
    return p.parse_args()

def main():
    proj = Path(__file__).resolve().parents[1]
    args = parse_args()

    csv_path  = (proj / args.csv).resolve()
    fasta_dir = (proj / args.fasta_dir).resolve()
    out_dir   = (proj / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Priority: CSV > FASTA directory
    records_csv  = read_csv(csv_path)
    records_fa   = read_single_fastas(fasta_dir)
    records: Dict[str, str] = dict(records_csv)  # copy
    for sid, seq in records_fa.items():
        if sid not in records:
            records[sid] = seq

    if not records:
        raise RuntimeError(f"No sequences found in {csv_path} or {fasta_dir}")

    use_templates = not args.no_templates
    jobs = build_jobs(
        records=records,
        use_templates=use_templates,
        max_template_date=args.max_template_date
    )

    # Write JSON (single or chunked)
    if args.chunk_size and args.chunk_size > 0:
        chunks = chunk_list(jobs, args.chunk_size)
        for i, part in enumerate(chunks, start=1):
            out_path = out_dir / f"{Path(args.outfile).stem}_part_{i}.json"
            out_path.write_text(json.dumps(part, indent=2), encoding="utf-8")
            print(f"[make_af3_jobs] wrote {len(part):4d} jobs -> {out_path}")
        print(f"[make_af3_jobs] total jobs: {len(jobs)}, files: {len(chunks)}")
    else:
        out_path = out_dir / args.outfile
        out_path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        print(f"[make_af3_jobs] wrote {len(jobs):4d} jobs -> {out_path}")

if __name__ == "__main__":
    main()
