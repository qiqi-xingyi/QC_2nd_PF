# --*-- conding:utf-8 --*--
# @time:9/10/25 01:59
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:clean_pdbbind_by_csv.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean PDBbind tree by keeping only IDs listed in data/seqs_len10.csv,
and remove all files/directories starting with '._' (macOS resource forks).

Default behavior: dry-run (print what would be removed). Use --apply to execute.

Usage examples (run from project root):
  # Dry-run (show plan only)
  python tools/clean_pdbbind_by_csv.py

  # Apply deletions
  python tools/clean_pdbbind_by_csv.py --apply

  # Also clean '._' files in extra locations (can repeat)
  python tools/clean_pdbbind_by_csv.py --apply --clean-root data/data_set --clean-root results
"""

from pathlib import Path
import argparse
import csv
import os
import shutil
from typing import Set, List, Tuple

def read_keep_ids(csv_path: Path) -> Set[str]:
    """Read 'id' column from CSV (utf-8-sig tolerant). Return lowercase IDs."""
    keep: Set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"No header found in CSV: {csv_path}")
        headers = {h.lower(): h for h in reader.fieldnames}
        id_key = headers.get("id")
        if not id_key:
            raise RuntimeError(f"'id' column is required in CSV: {csv_path}")
        for row in reader:
            sid = str(row[id_key]).strip().lower()
            if sid:
                keep.add(sid)
    if not keep:
        raise RuntimeError(f"No IDs parsed from CSV: {csv_path}")
    return keep

def list_immediate_subdirs(root: Path) -> List[Path]:
    """Return a list of immediate child directories under root."""
    return [p for p in sorted(root.iterdir()) if p.is_dir()]

def plan_pdbbind_deletions(pdbbind_root: Path, keep_ids: Set[str]) -> List[Path]:
    """Compute which subdirectories should be removed."""
    to_remove: List[Path] = []
    for sub in list_immediate_subdirs(pdbbind_root):
        pid = sub.name.strip().lower()
        # Ignore hidden or unexpected names starting with '._'
        if pid.startswith("._"):
            to_remove.append(sub)
            continue
        if pid not in keep_ids:
            to_remove.append(sub)
    return to_remove

def remove_dot_underscore(root: Path) -> Tuple[int, int]:
    """Recursively delete files/dirs whose name starts with '._'. Returns (files, dirs) removed."""
    files_removed = 0
    dirs_removed = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Remove offending directories first (modify dirnames in-place to avoid descending)
        bad_dirs = [d for d in dirnames if d.startswith("._")]
        for d in bad_dirs:
            full = Path(dirpath) / d
            try:
                shutil.rmtree(full)
                dirs_removed += 1
                # prevent walking into it
                dirnames.remove(d)
                print(f"[rm dir] {full}")
            except Exception as e:
                print(f"[warn] failed to remove dir {full}: {e}")
        # Remove offending files
        for fn in filenames:
            if fn.startswith("._"):
                full = Path(dirpath) / fn
                try:
                    full.unlink()
                    files_removed += 1
                    print(f"[rm file] {full}")
                except Exception as e:
                    print(f"[warn] failed to remove file {full}: {e}")
    return files_removed, dirs_removed

def main():
    ap = argparse.ArgumentParser(description="Keep only IDs from data/seqs_len10.csv and remove '._*' files.")
    ap.add_argument("--csv", type=str, default="data/seqs_len10.csv", help="CSV with 'id' column (default: data/seqs_len10.csv)")
    ap.add_argument("--pdbbind-root", type=str, default="data/Pdbbind", help="Root directory of PDBbind (default: data/Pdbbind)")
    ap.add_argument("--clean-root", action="append", default=[], help="Extra roots to clean '._' files (can repeat)")
    ap.add_argument("--apply", action="store_true", help="Actually perform deletions (default: dry-run)")
    args = ap.parse_args()

    proj = Path(".").resolve()
    csv_path = (proj / args.csv).resolve()
    pdbbind_root = (proj / args.pdbbind_root).resolve()

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not pdbbind_root.exists():
        raise SystemExit(f"PDBbind root not found: {pdbbind_root}")

    keep_ids = read_keep_ids(csv_path)
    print(f"[info] loaded {len(keep_ids)} keep IDs from {csv_path}")

    # 1) Plan deletions under PDBbind
    to_remove = plan_pdbbind_deletions(pdbbind_root, keep_ids)
    if to_remove:
        print(f"[plan] PDBbind subdirs to DELETE: {len(to_remove)}")
        for p in to_remove[:12]:
            print(f"  - {p}")
        if len(to_remove) > 12:
            print(f"  ... and {len(to_remove)-12} more")
    else:
        print("[plan] Nothing to delete under PDBbind (all subdirs are kept).")

    # 2) Clean '._' files/dirs under PDBbind and any extra roots
    clean_roots = [pdbbind_root] + [ (proj / r).resolve() for r in args.clean_root ]
    print(f"[plan] will clean '._*' under {len(clean_roots)} root(s):")
    for r in clean_roots:
        print(f"  - {r}")

    if not args.apply:
        print("\n[dry-run] No deletions performed. Re-run with --apply to execute.")
        return

    # --- APPLY ---
    # 2a) Delete unwanted PDBbind subdirectories
    removed_dirs = 0
    for d in to_remove:
        try:
            shutil.rmtree(d)
            removed_dirs += 1
            print(f"[deleted] {d}")
        except Exception as e:
            print(f"[warn] failed to delete {d}: {e}")
    print(f"[done] deleted PDBbind subdirs: {removed_dirs}")

    # 2b) Remove '._' files/dirs
    total_files_removed, total_dirs_removed = 0, 0
    for r in clean_roots:
        f_cnt, d_cnt = remove_dot_underscore(r)
        total_files_removed += f_cnt
        total_dirs_removed += d_cnt
    print(f"[done] removed '._*': files={total_files_removed}, dirs={total_dirs_removed}")

if __name__ == "__main__":
    main()
