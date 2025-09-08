# --*-- conding:utf-8 --*--
# @time:9/7/25 20:46
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:gen_angles_from_xyz.py

# tools/gen_angles_from_xyz.py
from __future__ import annotations
import os, math, glob
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q_ROOT = PROJECT_ROOT / "data" / "Quantum_original_data"

def dihedral(p0, p1, p2, p3) -> float:
    p0, p1, p2, p3 = [np.asarray(x, float) for x in (p0, p1, p2, p3)]
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return math.degrees(math.atan2(y, x))

def read_ca_xyz(xyz_path: Path):
    lines = [ln.strip() for ln in xyz_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    start = 1 if lines and lines[0].replace(".", "", 1).isdigit() else 0
    letters, coords = [], []
    for ln in lines[start:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        letters.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return letters, np.asarray(coords, float)

def virtual_phi_psi(coords: np.ndarray):
    n = len(coords)
    phi = np.full(n, np.nan)
    psi = np.full(n, np.nan)
    for i in range(n):
        if i-2 >= 0 and i+1 < n:
            phi[i] = dihedral(coords[i-2], coords[i-1], coords[i], coords[i+1])
        if i-1 >= 0 and i+2 < n:
            psi[i] = dihedral(coords[i-1], coords[i], coords[i+1], coords[i+2])
    return phi, psi  # 度

def process_one(pdbid: str, pdb_dir: Path):

    cand_files = []
    for i in range(1, 6):
        p1 = pdb_dir / f"{pdbid}_top_{i}.xyz"
        p2 = pdb_dir / f"{pdbid}_top_{i}"
        if p1.exists(): cand_files.append(("top_"+str(i), p1))
        elif p2.exists(): cand_files.append(("top_"+str(i), p2))
    if not cand_files:
        print(f"[WARN] no candidates in {pdb_dir}")
        return

    rows = []
    for cid, f in cand_files:
        letters, coords = read_ca_xyz(f)
        phi_deg, psi_deg = virtual_phi_psi(coords)
        # 输出一行：cid + 所有 phi_hat_* + 所有 psi_hat_*
        row = {"cid": cid}
        for i, ang in enumerate(phi_deg):
            row[f"phi_hat_{i}"] = ang
        for i, ang in enumerate(psi_deg):
            row[f"psi_hat_{i}"] = ang
        rows.append(row)

    df = pd.DataFrame(rows)
    out = pdb_dir / f"angles_{pdbid}.csv"
    df.to_csv(out, index=False)
    print(f"[OK] wrote {out}")

def main():
    for pdb_dir in sorted(Q_ROOT.iterdir()):
        if not pdb_dir.is_dir():
            continue
        pdbid = pdb_dir.name
        process_one(pdbid, pdb_dir)

if __name__ == "__main__":
    main()
