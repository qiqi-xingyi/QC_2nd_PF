# --*-- conding:utf-8 --*--
# @time:9/9/25 21:37
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:summarize_colabfold.py

"""
Summarize ColabFold outputs into a CSV:
  - id
  - best_pdb_path
  - mean_plddt (from PDB B-factor)
  - mean_pae (from pae.json or predicted_aligned_error*.json; optional)
  - n_atoms (count of ATOM records)
Output:
  results/colabfold_out/colabfold_master.csv
"""

import json
import re
import gzip
from pathlib import Path
import pandas as pd

def guess_best_pdb(dirpath: Path):
    for pat in ["relaxed_ranked_0.pdb", "ranked_0.pdb"]:
        p = dirpath / pat
        if p.exists():
            return p
    cands = sorted(dirpath.glob("ranked_*.pdb"))
    if cands:
        return cands[0]
    anyp = sorted(dirpath.glob("*.pdb"))
    return anyp[0] if anyp else None

def mean_plddt_from_pdb(pdb_path: Path):
    vals = []
    with pdb_path.open() as f:
        for ln in f:
            if ln.startswith("ATOM"):
                try:
                    b = float(ln[60:66])
                    vals.append(b)
                except Exception:
                    pass
    return (sum(vals) / len(vals)) if vals else None, len(vals)

def mean_pae_from_json(dirpath: Path):
    # Try a few common names; support gzip when present
    for name in ["pae.json", "predicted_aligned_error_v1.json", "pae.json.gz", "predicted_aligned_error_v1.json.gz"]:
        p = dirpath / name
        if p.exists():
            try:
                if p.suffix == ".gz":
                    with gzip.open(p, "rt", encoding="utf-8") as fh:
                        obj = json.load(fh)
                else:
                    obj = json.loads(p.read_text())
                mat = obj.get("pae") or obj.get("predicted_aligned_error")
                if isinstance(mat, list) and mat and isinstance(mat[0], list):
                    flat = [x for row in mat for x in row]
                    if flat:
                        return sum(flat) / len(flat)
            except Exception:
                pass
    return None

def summarize_dir(out_dir: Path, proj: Path) -> pd.DataFrame:
    rows = []
    subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
    files_flat = [p for p in out_dir.iterdir() if p.is_file()]

    if subdirs:
        for d in sorted(subdirs):
            sid = d.name
            pdb = guess_best_pdb(d)
            if not pdb:
                continue
            mean_plddt, n_atoms = mean_plddt_from_pdb(pdb)
            mean_pae = mean_pae_from_json(d)
            rows.append({
                "id": sid,
                "best_pdb_path": str(pdb.relative_to(proj)),
                "mean_plddt": mean_plddt,
                "mean_pae": mean_pae,
                "n_atoms": n_atoms
            })
    else:
        by_id = {}
        for f in files_flat:
            m = re.match(r"([A-Za-z0-9_:-]+)_ranked_\d+\.pdb$", f.name)
            if m:
                sid = m.group(1)
                by_id.setdefault(sid, []).append(f)
        for sid, plist in by_id.items():
            pdb = None
            for p in plist:
                if p.name.endswith("ranked_0.pdb"):
                    pdb = p; break
            if not pdb:
                pdb = sorted(plist)[0]
            mean_plddt, n_atoms = mean_plddt_from_pdb(pdb)
            mean_pae = mean_pae_from_json(out_dir)
            rows.append({
                "id": sid,
                "best_pdb_path": str(pdb.relative_to(proj)),
                "mean_plddt": mean_plddt,
                "mean_pae": mean_pae,
                "n_atoms": n_atoms
            })
    return pd.DataFrame(rows).sort_values("id")

def main():
    proj = Path(__file__).resolve().parents[1]
    out_dir = proj / "results" / "colabfold_out"
    out_csv = out_dir / "colabfold_master.csv"

    if not out_dir.exists():
        print(f"[summarize_colabfold] output dir not found: {out_dir}")
        return

    df = summarize_dir(out_dir, proj)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[summarize_colabfold] saved {len(df)} records → {out_csv}")

if __name__ == "__main__":
    main()

