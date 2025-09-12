# --*-- coding:utf-8 --*--
# @time: 9/7/25
# @Author : Yuqi Zhang
# @Email  : yzhan135@kent.edu
# @File   : rerank.py

"""
E_rerank.rerank (final, with SS3/SS8 fusion):

What this module does
---------------------
- Robustly reads quantum_rmsd candidates & energies (supports "Rank i: <float>" text).
- Reads NetSurfP priors (headerless TSV/CSV, 19-column standard layout).
- For each candidate (top_1..top_N):
    * Load per-structure angles from angles_<pdbid>.csv if present (degrees).
    * Otherwise, compute "virtual" phi/psi (radians) from C-alpha-only xyz.
    * Derive candidate SS3/SS8 probabilities from angles (Ramachandran kernels).
- Calls FusionReRanker to combine:
    * quantum_rmsd energy (E_q)
    * secondary-structure distance (D_ss)
    * angle difference (D_{phi,psi}), optionally RSA-weighted
- Writes:
    * rerank.csv (score breakdown)
    * metadata.json (parameters)
    * copies & renames xyz files by new ranking order

Notes
-----
- RSA column in NSP is clipped to [0,1]; missing/NaN -> uniform weights (=1.0).
- Edge residues have undefined dihedrals with C-alpha-only xyz (NaN by definition);
  FusionReRanker masks them out when averaging.
"""


from __future__ import annotations
import os
import json
import math
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # Prefer package-relative import
    from .fusion_reranker import FusionReRanker, Candidate
except Exception:  # pragma: no cover
    # Fallback for direct execution
    import sys
    sys.path.append(os.getcwd())
    from fusion_reranker import FusionReRanker, Candidate


# ---------------------------- Small data holders ----------------------------
@dataclass
class Paths:
    quantum_root: str
    nsp_root: str
    out_root: str


# ---------------------------- Main orchestrator -----------------------------
class ERerank:
    def __init__(
        self,
        quantum_root: str,
        nsp_root: str,
        out_root: str,
        ss_mode: str = "ss3",          # or "ss8"
        dist: str = "ce",               # "ce" | "kl" | "l2"
        angle_weight: str = "rsa",      # "rsa" | "uniform"
        normalize_terms: bool = True,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        max_candidates: int = 5,
        random_seed: int = 0,
    ) -> None:
        self.paths = Paths(quantum_root, nsp_root, out_root)
        self.ss_mode = ss_mode
        self.dist = dist
        self.angle_weight = angle_weight
        self.normalize_terms = normalize_terms
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.max_candidates = max_candidates
        self.rng = np.random.default_rng(random_seed)
        os.makedirs(out_root, exist_ok=True)

    # ---------------------------- Index reading ----------------------------
    def _read_index_pdbids(self, index_file: str) -> List[str]:
        """
        Read pdbids from index file in a simple and robust way (no pandas guessing).
        The file is expected to have a header line 'pdbid' followed by one id per line.
        Any BOM or blank lines are ignored.
        """
        ids = []
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.lower().replace("\ufeff", "") == "pdbid":
                    continue
                ids.append(s)
        return ids

    def run(self, index_file: str) -> None:
        """
        Process all PDBIDs listed in index_file:
        - filter out those without quantum_rmsd dir or NSP tsv/csv
        - process each valid pdbid
        """
        pdbids = self._read_index_pdbids(index_file)
        print(f"[INFO] Loaded {len(pdbids)} pdbids from index. Sample: {', '.join(pdbids[:8])}")

        valid: List[str] = []
        for pid in pdbids:
            pid_norm = pid.strip()
            q_dir = os.path.join(self.paths.quantum_root, pid_norm)
            nsp_tsv = os.path.join(self.paths.nsp_root, f"{pid_norm}.tsv")
            nsp_csv = os.path.join(self.paths.nsp_root, f"{pid_norm}.csv")

            if not os.path.isdir(q_dir):
                print(f"[WARN] Skip {pid_norm}: quantum_rmsd dir missing -> {q_dir}")
                continue
            if not (os.path.isfile(nsp_tsv) or os.path.isfile(nsp_csv)):
                print(f"[WARN] Skip {pid_norm}: NSP file missing -> {nsp_tsv} / {nsp_csv}")
                continue

            valid.append(pid_norm)

        if not valid:
            raise RuntimeError("No valid pdbids after filtering. Check your data paths.")

        for pdbid in valid:
            self._process_one(pdbid)

    def _process_one(self, pdbid: str) -> None:
        """
        Process a single pdbid:
        - read quantum_rmsd candidates and energies
        - read NSP priors
        - build candidate list (ensure each candidate carries phi/psi and ss_probs_hat)
        - rerank and materialize outputs
        """
        q_dir = os.path.join(self.paths.quantum_root, pdbid)
        out_dir = os.path.join(self.paths.out_root, pdbid)
        os.makedirs(out_dir, exist_ok=True)

        energies, cand_map = self._read_quantum_candidates(q_dir, pdbid)
        priors = self._read_nsp_priors(pdbid)
        cands = self._build_candidates(pdbid, q_dir, energies, cand_map)
        df_rank = self._rerank(priors, cands)
        self._materialize_outputs(pdbid, q_dir, out_dir, df_rank, cand_map)

    # ---------------------------- Quantum IO helpers ----------------------------
    def _read_quantum_candidates(self, q_dir: str, pdbid: str) -> Tuple[List[float], Dict[str, str]]:
        """
        Discover candidate xyz files (with or without '.xyz') and align them with energies.
        Energies are read from a variety of formats, including 'Rank i: value' text.
        """
        if not os.path.isdir(q_dir):
            raise FileNotFoundError(f"Quantum directory not found: {q_dir}")

        # Discover candidate files in order top_1..top_N
        cand_map: Dict[str, str] = {}
        for i in range(1, 100):
            p1 = os.path.join(q_dir, f"{pdbid}_top_{i}.xyz")
            p2 = os.path.join(q_dir, f"{pdbid}_top_{i}")
            if os.path.isfile(p1):
                cand_map[f"top_{i}"] = p1
            elif os.path.isfile(p2):
                cand_map[f"top_{i}"] = p2
            else:
                break
        if not cand_map:
            raise FileNotFoundError(f"No candidate files like {pdbid}_top_1(.xyz) under {q_dir}")

        # Enforce max_candidates
        cand_map = dict(list(sorted(cand_map.items(), key=lambda x: int(x[0].split('_')[-1])))[: self.max_candidates])

        # Try known energy filenames, then any csv/tsv/txt with 'energy' in its name
        energy_files = [
            os.path.join(q_dir, f"top_5_energies_{pdbid}.txt"),
            os.path.join(q_dir, f"energy_list_{pdbid}.txt"),
            os.path.join(q_dir, f"energies_{pdbid}.txt"),
        ]
        energies: Optional[List[float]] = None
        for ef in energy_files:
            if os.path.isfile(ef):
                energies = self._read_energy_file(ef)
                break
        if energies is None:
            for fn in os.listdir(q_dir):
                if fn.lower().endswith((".csv", ".tsv", ".txt")) and "energy" in fn.lower():
                    energies = self._read_energy_file(os.path.join(q_dir, fn))
                    break
        if energies is None:
            raise FileNotFoundError(f"Cannot find an energy file in {q_dir}")

        if len(energies) < len(cand_map):
            raise ValueError(f"Energy list shorter ({len(energies)}) than candidates ({len(cand_map)})")
        energies = energies[: len(cand_map)]
        return energies, cand_map

    def _read_energy_file(self, path: str) -> List[float]:
        """
        Supported formats:
        - CSV/TSV with a column named like 'energy'
        - Plain text with one value per line
        - Lines like: 'Rank 1: 6592.7348' (extract the last float per line)
        """
        import re
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".tsv"):
            try:
                df = pd.read_csv(path, sep=None, engine="python")
                for col in df.columns:
                    if col.lower() in {"energy", "energies", "e", "ene", "value"}:
                        return df[col].astype(float).tolist()
            except Exception:
                pass

        vals: List[float] = []
        float_re = re.compile(r"[-+]?((\d+\.?\d*)|(\.\d+))([eE][-+]?\d+)?")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                m = list(float_re.finditer(s))
                if m:
                    try:
                        vals.append(float(m[-1].group(0)))  # last float on the line
                        continue
                    except ValueError:
                        pass
                # Fallback: try the first token
                tok = s.split()[0]
                try:
                    vals.append(float(tok))
                except ValueError:
                    continue
        if not vals:
            raise ValueError(f"No energies parsed from {path}")
        return vals

    # ---------------------------- NSP priors ----------------------------
    def _read_nsp_priors(self, pdbid: str) -> Dict[str, np.ndarray]:
        """
        Read NetSurfP predictions (headerless TSV/CSV).
        Expected 19 columns layout:
          0: idx, 1: aa,
          2..9:  ss8 (8 cols)
          10..12:ss3 (3 cols)
          13..14:disorder (2 cols)
          15: RSA, 16: ASA,
          17: phi(deg), 18: psi(deg)
        If columns are fewer, a robust fallback tries to locate phi/psi and an RSA-like [0,1] column.
        """
        # Locate file
        tsv_path = None
        for cand in (f"{pdbid}.tsv", f"{pdbid}.csv"):
            p = os.path.join(self.paths.nsp_root, cand)
            if os.path.isfile(p):
                tsv_path = p
                break
        if tsv_path is None:
            raise FileNotFoundError(f"NSP3 tsv/csv for {pdbid} not found under {self.paths.nsp_root}")

        df = pd.read_csv(tsv_path, sep=None, engine="python", header=None)
        num = df.apply(pd.to_numeric, errors="coerce")

        # Prefer the standard 19-column layout
        if num.shape[1] >= 19:
            ss8 = num.iloc[:, 2:10].to_numpy(float)
            ss3 = num.iloc[:, 10:13].to_numpy(float)
            rsa = num.iloc[:, 15].to_numpy(float)
            phi = num.iloc[:, 17].to_numpy(float)
            psi = num.iloc[:, 18].to_numpy(float)
        else:
            # Fallback: infer phi/psi as the last two numeric columns; infer RSA as a [0,1] column to the left of phi.
            valid_cols = [i for i in range(num.shape[1]) if num.iloc[:, i].notna().any()]
            if len(valid_cols) < 2:
                raise ValueError("Cannot locate numeric columns for phi/psi in NSP file")
            psi_col = valid_cols[-1]
            phi_col = valid_cols[-2]

            def safe_slice(a, b):
                b = min(b, num.shape[1])
                if a >= b:
                    return np.zeros((len(num), max(0, b - a)))
                return num.iloc[:, a:b].to_numpy(float)

            ss8 = safe_slice(2, 10)
            ss3 = safe_slice(10, 13)

            rsa = None
            for c in range(phi_col - 1, 1, -1):
                col = num.iloc[:, c].to_numpy(float)
                finite = col[np.isfinite(col)]
                if finite.size == 0:
                    continue
                if finite.min() >= -1e-6 and finite.max() <= 1.0 + 1e-6:
                    rsa = col
                    break
            phi = num.iloc[:, phi_col].to_numpy(float)
            psi = num.iloc[:, psi_col].to_numpy(float)

        # Choose SS3 or SS8 according to config and normalize rows
        ss_probs = ss3 if self.ss_mode == "ss3" else ss8
        ss_probs = np.nan_to_num(ss_probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = ss_probs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        ss_probs = ss_probs / row_sum

        # Convert angles to radians
        phi = np.deg2rad(phi)
        psi = np.deg2rad(psi)

        # Clip RSA into [0,1]; replace NaN by 1.0 (uniform weight) to avoid poisoning averages.
        rsa_arr = None
        if rsa is not None:
            rsa_arr = np.asarray(rsa, float)
            rsa_arr = np.where(np.isfinite(rsa_arr), np.clip(rsa_arr, 0.0, 1.0), 1.0)

        return {
            "ss_probs": ss_probs,
            "phi": phi,
            "psi": psi,
            "rsa": rsa_arr,
        }

    # ---------------------------- XYZ -> virtual angles ----------------------------
    def _dihedral(self, p0, p1, p2, p3) -> float:
        """
        Compute dihedral angle (radians) of four points using the right-hand rule.
        Using C-alpha-only chain to form a "virtual" backbone dihedral.
        """
        p0, p1, p2, p3 = [np.asarray(x, float) for x in (p0, p1, p2, p3)]
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2
        b1n = b1 / (np.linalg.norm(b1) + 1e-12)
        v = b0 - np.dot(b0, b1n) * b1n
        w = b2 - np.dot(b2, b1n) * b1n
        x = np.dot(v, w)
        y = np.dot(np.cross(b1n, v), w)
        return math.atan2(y, x)  # [-pi, pi]

    def _virtual_angles_from_xyz(self, xyz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a C-alpha-only xyz file and compute per-residue "virtual" phi/psi (radians).
        Edges will be NaN because 4 points are required to define a dihedral.
        """
        with open(xyz_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        # Skip the first line if it is a numeric atom count (XYZ header)
        start = 1 if lines and lines[0].replace(".", "", 1).isdigit() else 0

        coords: List[List[float]] = []
        for ln in lines[start:]:
            parts = ln.split()
            if len(parts) < 4:
                continue
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        coords = np.asarray(coords, float)
        n = len(coords)
        phi = np.full(n, np.nan, dtype=float)
        psi = np.full(n, np.nan, dtype=float)

        for i in range(n):
            if i - 2 >= 0 and i + 1 < n:
                phi[i] = self._dihedral(coords[i - 2], coords[i - 1], coords[i], coords[i + 1])
            if i - 1 >= 0 and i + 2 < n:
                psi[i] = self._dihedral(coords[i - 1], coords[i], coords[i + 1], coords[i + 2])

        return phi, psi

    # ---------------------------- Angles -> SS3/SS8 probabilities ----------------------------
    def _angles_to_ss_probs(self, phi: np.ndarray, psi: np.ndarray, mode: str = "ss3") -> np.ndarray:
        """
        Map per-residue (phi, psi) to a secondary-structure probability vector.

        Step 1: compute SS3 probs (H/E/C) via Gaussian kernels centered on typical
                Ramachandran regions; sigma can be tuned.
        Step 2: if mode == 'ss8', expand SS3 to SS8 with a fixed heuristic mapping.

        Returns
        -------
        probs : (L, K) ndarray
            K=3 for ss3, K=8 for ss8
        """
        # centers in degrees: (phi, psi)
        centers = {
            "H": (-60.0, -45.0),   # alpha-helix basin
            "E": (-120.0, 130.0),  # beta basin
            "C": (0.0, 0.0),       # coil/other (broad catch-all)
        }
        sigma = 40.0  # std in degrees for both phi and psi
        phi_d = np.rad2deg(phi)
        psi_d = np.rad2deg(psi)

        def circ_dist(a, b):
            """circular distance in degrees, in [0, 180]"""
            d = abs(a - b) % 360.0
            return min(d, 360.0 - d)

        L = len(phi_d)
        ss3 = np.zeros((L, 3), dtype=float)
        for i in range(L):
            if not (np.isfinite(phi_d[i]) and np.isfinite(psi_d[i])):
                ss3[i] = [1/3, 1/3, 1/3]
                continue
            scores = []
            for key in ("H", "E", "C"):
                mu_phi, mu_psi = centers[key]
                dphi = circ_dist(phi_d[i], mu_phi)
                dpsi = circ_dist(psi_d[i], mu_psi)
                s = math.exp(-0.5 * ((dphi/sigma)**2 + (dpsi/sigma)**2))
                scores.append(s)
            scores = np.asarray(scores)
            scores = scores / (scores.sum() + 1e-12)
            ss3[i] = scores  # order: H, E, C

        if mode == "ss3":
            return ss3

        # ss8 expansion (heuristic): distribute H into H/G/I; E into E/B; C into T/S/L.
        # Order for ss8 probs here: [H, G, I, E, B, T, S, L]
        ss8 = np.zeros((L, 8), dtype=float)
        for i in range(L):
            pH, pE, pC = ss3[i]
            h_split = np.array([0.8, 0.1, 0.1])  # H->(H,G,I)
            e_split = np.array([0.9, 0.1])       # E->(E,B)
            c_split = np.array([1/3, 1/3, 1/3])  # C->(T,S,L)
            ss8[i, 0:3] = pH * h_split
            ss8[i, 3:5] = pE * e_split
            ss8[i, 5:8] = pC * c_split
        return ss8

    # ---------------------------- Build candidates ----------------------------
    def _build_candidates(
        self,
        pdbid: str,
        q_dir: str,
        energies: List[float],
        cand_map: Dict[str, str],
    ) -> List[Candidate]:
        """
        Build Candidate objects. If angles_<pdbid>.csv exists, try to use it.
        Otherwise (or if malformed) compute virtual angles from each xyz as fallback.
        In both cases, derive candidate SS probs (SS3/SS8) from angles to enable D_ss.
        """
        ang_path = os.path.join(q_dir, f"angles_{pdbid}.csv")
        angles_df: Optional[pd.DataFrame] = None
        if os.path.isfile(ang_path):
            angles_df = pd.read_csv(ang_path)
            if "cid" not in angles_df.columns:
                raise ValueError(f"angles file missing 'cid' column: {ang_path}")

        cands: List[Candidate] = []
        order = sorted(cand_map.keys(), key=lambda x: int(x.split('_')[-1]))
        for i, cid in enumerate(order):
            E_q = float(energies[i])
            phi_hat: Optional[np.ndarray] = None
            psi_hat: Optional[np.ndarray] = None

            # 1) Try angles_<pdbid>.csv first (degrees)
            if angles_df is not None:
                row = angles_df.loc[angles_df["cid"] == cid]
                if len(row) == 1:
                    row = row.iloc[0]
                    phi_cols = [c for c in angles_df.columns if c.startswith("phi_hat_")]
                    psi_cols = [c for c in angles_df.columns if c.startswith("psi_hat_")]
                    if phi_cols and psi_cols:
                        phi_hat = np.deg2rad(row[phi_cols].to_numpy(float))
                        psi_hat = np.deg2rad(row[psi_cols].to_numpy(float))

            # 2) Fallback: compute "virtual" angles from xyz if file missing / malformed / empty
            if phi_hat is None or psi_hat is None or len(phi_hat) == 0 or len(psi_hat) == 0:
                phi_hat, psi_hat = self._virtual_angles_from_xyz(cand_map[cid])

            # 3) Derive candidate SS probs from angles (Ramachandran-based)
            ss_hat = self._angles_to_ss_probs(phi_hat, psi_hat, mode=self.ss_mode)

            cands.append(Candidate(
                cid=cid,
                E_q=E_q,
                phi_hat=phi_hat,
                psi_hat=psi_hat,
                ss_probs_hat=ss_hat,   # <-- enable D_ss
                unit="rad",
                meta={"xyz_path": cand_map[cid]},
            ))
        return cands

    # ---------------------------- Rerank & write outputs ----------------------------
    def _rerank(self, priors: Dict[str, np.ndarray], cands: List[Candidate]) -> pd.DataFrame:
        """
        Rerank candidates using FusionReRanker which combines:
        - quantum_rmsd energy
        - secondary structure distance (if candidate supplies ss_probs_hat)
        - angle differences (phi/psi), weighted by RSA if available
        """
        rer = FusionReRanker(
            ss_mode=self.ss_mode,
            dist=self.dist,
            angle_weight=self.angle_weight,
            normalize_terms=self.normalize_terms,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )
        rer.load_priors(
            ss_probs=priors["ss_probs"],
            phi=priors["phi"],
            psi=priors["psi"],
            rsa=priors.get("rsa"),
            unit="rad",
        )
        return rer.rerank(cands)

    def _materialize_outputs(
        self,
        pdbid: str,
        q_dir: str,
        out_dir: str,
        df_rank: pd.DataFrame,
        cand_map: Dict[str, str],
    ) -> None:
        """
        Save:
        - rerank.csv (score breakdown)
        - metadata.json (parameters & notes)
        - copy/rename xyz files by new ranking order
        """
        df_rank.to_csv(os.path.join(out_dir, "rerank.csv"), index=False)

        for new_idx, row in df_rank.reset_index(drop=True).iterrows():
            cid = row["cid"]
            src = cand_map[cid]
            new_name = f"{new_idx + 1:02d}_{os.path.basename(src)}"
            shutil.copy2(src, os.path.join(out_dir, new_name))

        meta = {
            "pdbid": pdbid,
            "quantum_dir": q_dir,
            "params": {
                "ss_mode": self.ss_mode,
                "dist": self.dist,
                "angle_weight": self.angle_weight,
                "normalize_terms": self.normalize_terms,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "max_candidates": self.max_candidates,
            },
            "notes": [
                "If candidate angles are absent, virtual phi/psi are computed from C-alpha-only xyz.",
                "RSA is used to weight angle differences when available; missing RSA -> uniform weights.",
                "Candidate SS3/SS8 probabilities are derived from angles via Ramachandran kernels.",
            ],
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


# ---------------------------- Optional CLI ----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="E_rerank (final, with SS3/SS8 fusion)")
    ap.add_argument("--quantum_root", required=True)
    ap.add_argument("--nsp_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--index_file", required=True)
    ap.add_argument("--ss_mode", choices=["ss3", "ss8"], default="ss3")
    ap.add_argument("--dist", choices=["l2", "kl", "ce"], default="ce")
    ap.add_argument("--angle_weight", choices=["uniform", "rsa"], default="rsa")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--max_candidates", type=int, default=5)

    args = ap.parse_args()

    r = ERerank(
        quantum_root=args.quantum_root,
        nsp_root=args.nsp_root,
        out_root=args.out_root,
        ss_mode=args.ss_mode,
        dist=args.dist,
        angle_weight=args.angle_weight,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        max_candidates=args.max_candidates,
    )
    r.run(index_file=args.index_file)
