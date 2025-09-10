# --*-- conding:utf-8 --*--
# @time:9/10/25 01:46
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:filter_and_merge_pdbbind.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter PDBbind dataset by a 'keep' list (predicted IDs) and merge scattered
dataset *.txt entries into a single CSV, while removing/archiving extra structures.

Inputs (all optional; you can pass multiple sources):
  --keep-csv PATH            CSV with column 'id' (optionally 'sequence')
  --keep-fasta-dir PATH      Directory with single-entry FASTA files (*.fasta)
  --keep-manifest PATH       CSV manifest (e.g., master_ai_colabfold.csv with 'id')
  --keep-dir-names PATH      Directory whose immediate subfolders are IDs (e.g., results/colabfold_web)

Dataset & structures:
  --dataset-dir data/data_set
  --pdbbind-root data/Pdbbind

Filtering:
  --min-len 11               Only keep entries with length >= min_len
                             (length inferred from sequence string if present, else end-start+1)

Cleanup mode:
  --apply                    Actually apply cleanup (default is dry-run)
  --remove-mode archive|delete  'archive' (default) moves extra IDs to data/Pdbbind_removed/
                                'delete' permanently removes them (use with caution)

Outputs:
  - data/data_set/filtered_merged.csv
  - data/data_set/keep_ids.txt (sorted)
  - (optional) data/Pdbbind_removed/<id>/ ... when archiving

Example:
  python tools/filter_and_merge_pdbbind.py \
    --keep-csv data/seqs_len10.csv \
    --keep-fasta-dir data/seqs_len10 \
    --keep-manifest results/colabfold_web/master_ai_colabfold.csv \
    --keep-dir-names results/colabfold_web \
    --min-len 11 --apply
"""

from pathlib import Path
import argparse
import csv
import re
import shutil
from typing import Dict, List, Set, Tuple

# --- 20 standard amino acids 3->1 mapping
AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}
AA1_SET = set(AA3_TO_1.values())

def clean_id(s: str) -> str:
    return (s or "").strip().lower()

def clean_seq1(s: str) -> str:
    s = (s or "").strip().upper().replace(" ", "")
    return "".join([c for c in s if c in AA1_SET or c==":"])

def seq3_to_seq1(seq3_hyphen: str) -> str:
    if not seq3_hyphen:
        return ""
    parts = [p.strip().upper() for p in seq3_hyphen.split("-") if p.strip()]
    out = []
    for p in parts:
        out.append(AA3_TO_1.get(p,"X"))
    return "".join(out)

def read_keep_from_csv(path: Path) -> Set[str]:
    keep: Set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return keep
        headers = {h.lower(): h for h in reader.fieldnames}
        id_key = headers.get("id")
        if not id_key:
            return keep
        for row in reader:
            sid = clean_id(row[id_key])
            if sid:
                keep.add(sid)
    return keep

def read_keep_from_fasta_dir(d: Path) -> Set[str]:
    keep: Set[str] = set()
    if not d.exists(): return keep
    for fa in sorted(d.glob("*.fasta")):
        # use filename as id (strip extension), or header if needed
        sid = clean_id(fa.stem)
        if sid:
            keep.add(sid)
    return keep

def read_keep_from_manifest(path: Path) -> Set[str]:
    # CSV with at least 'id' column
    return read_keep_from_csv(path)

def read_keep_from_dirnames(d: Path) -> Set[str]:
    keep: Set[str] = set()
    if not d.exists(): return keep
    for p in d.iterdir():
        if p.is_dir():
            keep.add(clean_id(p.name))
    return keep

def merge_keep_sets(sets: List[Set[str]]) -> Set[str]:
    out: Set[str] = set()
    for s in sets:
        out.update([clean_id(x) for x in s])
    return out

def parse_dataset_txt(txt_path: Path) -> List[dict]:
    """Parse a dataset .txt file; return list of dicts with fields below."""
    rows: List[dict] = []
    # Patterns
    pat_chain = re.compile(r"Chain\s+([A-Za-z0-9])", re.IGNORECASE)
    pat_res   = re.compile(r"Residues\s+(\d+)-(\d+)", re.IGNORECASE)
    # We also try to capture the trailing 3-letter sequence segment
    for ln in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = re.split(r"\s*\t\s*|\s{2,}", ln)
        if len(parts) < 2:
            parts = ln.split()
        if not parts:
            continue
        pid = clean_id(parts[0])
        pocket_file = parts[1] if len(parts) > 1 else None

        m_chain = pat_chain.search(ln)
        m_res   = pat_res.search(ln)

        chain = m_chain.group(1) if m_chain else None
        start = int(m_res.group(1)) if m_res else None
        end   = int(m_res.group(2)) if m_res else None

        # Try to split out the last column as 3-letter sequence if present
        seq3 = None
        if len(parts) >= 5:
            # Many of your lines look like: id pocket "Chain A" "Residues s-e" "GLY-ASN-..."
            seq3 = parts[-1]

        rows.append({
            "id": pid,
            "pocket_file": pocket_file,
            "chain": chain,
            "start": start,
            "end": end,
            "seq3": seq3,
            "txt": str(txt_path)
        })
    return rows

def load_dataset_indices(dataset_dir: Path) -> List[dict]:
    all_rows: List[dict] = []
    for txt in sorted(dataset_dir.glob("*.txt")):
        all_rows.extend(parse_dataset_txt(txt))
    # Deduplicate by (id); keep the first occurrence
    seen = set()
    uniq = []
    for r in all_rows:
        k = r["id"]
        if k and k not in seen:
            uniq.append(r); seen.add(k)
    return uniq

def ensure_archive_dir(root: Path) -> Path:
    arc = root.parent / (root.name + "_removed")
    arc.mkdir(parents=True, exist_ok=True)
    return arc

def main():
    ap = argparse.ArgumentParser(description="Filter PDBbind by predicted IDs and merge dataset entries; remove extra structures.")
    ap.add_argument("--dataset-dir", type=str, default="data/data_set")
    ap.add_argument("--pdbbind-root", type=str, default="data/Pdbbind")
    ap.add_argument("--out-merged", type=str, default="data/data_set/filtered_merged.csv")
    ap.add_argument("--out-keep", type=str, default="data/data_set/keep_ids.txt")
    ap.add_argument("--min-len", type=int, default=11, help="Minimum residue length to keep (default 11)")
    # keep sources
    ap.add_argument("--keep-csv", action="append", default=[], help="CSV with 'id' column (e.g., data/seqs_len10.csv)")
    ap.add_argument("--keep-fasta-dir", action="append", default=[], help="Directory with *.fasta")
    ap.add_argument("--keep-manifest", action="append", default=[], help="Manifest CSV with 'id' (e.g., master_ai_colabfold.csv)")
    ap.add_argument("--keep-dir-names", action="append", default=[], help="Directory whose subfolders are IDs (e.g., results/colabfold_web)")
    # cleanup
    ap.add_argument("--apply", action="store_true", help="Apply cleanup (otherwise dry-run)")
    ap.add_argument("--remove-mode", choices=["archive","delete"], default="archive",
                    help="archive (move to Pdbbind_removed) or delete permanently")
    args = ap.parse_args()

    proj = Path(".").resolve()
    dataset_dir = (proj / args.dataset_dir).resolve()
    pdbbind_root = (proj / args.pdbbind_root).resolve()
    out_merged = (proj / args.out_merged).resolve()
    out_keep = (proj / args.out_keep).resolve()

    # 1) Build keep set
    keep_sets: List[Set[str]] = []
    for p in args.keep_csv:
        keep_sets.append(read_keep_from_csv(Path(p)))
    for p in args.keep_fasta_dir:
        keep_sets.append(read_keep_from_fasta_dir(Path(p)))
    for p in args.keep_manifest:
        keep_sets.append(read_keep_from_manifest(Path(p)))
    for p in args.keep_dir_names:
        keep_sets.append(read_keep_from_dirnames(Path(p)))

    keep_ids = sorted(merge_keep_sets(keep_sets))
    if not keep_ids:
        raise SystemExit("No keep IDs collected. Provide at least one --keep-* source.")
    out_keep.parent.mkdir(parents=True, exist_ok=True)
    Path(out_keep).write_text("\n".join(keep_ids) + "\n", encoding="utf-8")
    print(f"[info] keep IDs: {len(keep_ids)} -> {out_keep}")

    keep_set = set(keep_ids)

    # 2) Load dataset entries and filter to keep_ids and min length
    entries = load_dataset_indices(dataset_dir)
    merged_rows: List[dict] = []

    for r in entries:
        pid = r["id"]
        if not pid or pid not in keep_set:
            continue
        start, end, seq3 = r["start"], r["end"], r["seq3"]
        # length by seq3 if present, else by range
        if seq3 and "-" in seq3:
            seq1 = seq3_to_seq1(seq3)
            length = len(seq1.replace(":", ""))
        else:
            seq1 = ""
            if start is not None and end is not None and end >= start:
                length = end - start + 1
            else:
                length = 0
        if length < args.min_len:
            continue

        merged_rows.append({
            "id": pid,
            "pocket_file": r["pocket_file"],
            "chain": r["chain"],
            "start": start,
            "end": end,
            "length": length,
            "seq3": seq3,
            "seq1": seq1
        })

    # Write merged CSV
    import pandas as pd
    out_merged.parent.mkdir(parents=True, exist_ok=True)
    if merged_rows:
        df = pd.DataFrame(merged_rows).sort_values("id")
        df.to_csv(out_merged, index=False)
        print(f"[info] merged entries: {len(df)} -> {out_merged}")
    else:
        print("[warn] no merged rows matched keep set and length filter.")

    # 3) Cleanup PDBbind: remove/archive directories NOT in keep_set
    if pdbbind_root.exists():
        to_remove: List[Path] = []
        for p in pdbbind_root.iterdir():
            if not p.is_dir():
                continue
            pid = clean_id(p.name)
            # ignore macOS metadata dirs (e.g., names starting with '._') just in case
            if pid.startswith("._"):
                continue
            if pid not in keep_set:
                to_remove.append(p)

        if not to_remove:
            print("[info] no extra PDBbind entries to remove.")
        else:
            print(f"[plan] extra PDBbind entries: {len(to_remove)}")
            if args.apply:
                if args.remove_mode == "archive":
                    arc = ensure_archive_dir(pdbbind_root)
                    for d in to_remove:
                        dest = arc / d.name
                        # avoid overwriting: add suffix if exists
                        i, final = 1, dest
                        while final.exists():
                            final = arc / f"{d.name}__{i}"
                            i += 1
                        shutil.move(str(d), str(final))
                        print(f"[archived] {d} -> {final}")
                else:  # delete
                    for d in to_remove:
                        shutil.rmtree(d)
                        print(f"[deleted] {d}")
            else:
                for d in to_remove[:10]:
                    print(f"  would remove: {d}")
                if len(to_remove) > 10:
                    print(f"  ... and {len(to_remove)-10} more (dry-run)")
                print("Run again with --apply to execute.")
    else:
        print(f"[warn] PDBbind root not found: {pdbbind_root}")

if __name__ == "__main__":
    main()
