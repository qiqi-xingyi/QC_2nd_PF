# --*-- conding:utf-8 --*--
# @time:9/7/25 19:14
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:rerank.py

"""
E_rerank.rerank (fixed):
- Robust energy parser supports lines like: "Rank 1: 6592.7348".
- Candidate discovery accepts files with or without the .xyz extension.

Drop-in replacement for E_rerank/rerank.py
"""

from __future__ import annotations
import os
import json
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .fusion_reranker import FusionReRanker, Candidate
except Exception:  # pragma: no cover
    import sys
    sys.path.append(os.getcwd())
    from fusion_reranker import FusionReRanker, Candidate


@dataclass
class Paths:
    quantum_root: str
    nsp_root: str
    out_root: str


class ERerank:
    def __init__(
        self,
        quantum_root: str,
        nsp_root: str,
        out_root: str,
        ss_mode: str = "ss3",
        dist: str = "ce",
        angle_weight: str = "rsa",
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

    def _read_index_pdbids(self, index_file: str):
        """Read pdbids from index file robustly (no pandas, no guessing)."""
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

    # 用下面这个 run() 替换原来的 run()
    def run(self, index_file: str) -> None:
        """Process all PDBIDs listed in index_file."""
        pdbids = self._read_index_pdbids(index_file)

        print(f"[INFO] Loaded {len(pdbids)} pdbids from index. Sample: {', '.join(pdbids[:8])}")

        valid = []
        for pid in pdbids:
            pid_norm = pid.strip()
            q_dir = os.path.join(self.paths.quantum_root, pid_norm)
            nsp_tsv = os.path.join(self.paths.nsp_root, f"{pid_norm}.tsv")
            nsp_csv = os.path.join(self.paths.nsp_root, f"{pid_norm}.csv")

            if not os.path.isdir(q_dir):
                print(f"[WARN] Skip {pid_norm}: quantum dir missing -> {q_dir}")
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
        q_dir = os.path.join(self.paths.quantum_root, pdbid)
        out_dir = os.path.join(self.paths.out_root, pdbid)
        os.makedirs(out_dir, exist_ok=True)

        energies, cand_map = self._read_quantum_candidates(q_dir, pdbid)
        priors = self._read_nsp_priors(pdbid)
        cands = self._build_candidates(pdbid, q_dir, energies, cand_map)
        df_rank = self._rerank(priors, cands)
        self._materialize_outputs(pdbid, q_dir, out_dir, df_rank, cand_map)

    # ---------------------------- IO helpers ----------------------------
    def _read_quantum_candidates(self, q_dir: str, pdbid: str) -> Tuple[List[float], Dict[str, str]]:
        if not os.path.isdir(q_dir):
            raise FileNotFoundError(f"Quantum directory not found: {q_dir}")

        # discover files: accept with or without .xyz extension
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

        cand_map = dict(list(sorted(cand_map.items(), key=lambda x: int(x[0].split('_')[-1])))[: self.max_candidates])

        # energy files
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
            # generic csv/tsv/txt with 'energy' in filename
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
        """Supported formats:
        - CSV/TSV with a column named like 'energy'
        - Plain text with one value per line
        - Lines like: "Rank 1: 6592.7348" (extract last float per line)
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
        # text parsing
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
                        vals.append(float(m[-1].group(0)))
                        continue
                    except ValueError:
                        pass
                # fallback: first token
                tok = s.split()[0]
                try:
                    vals.append(float(tok))
                except ValueError:
                    continue
        if not vals:
            raise ValueError(f"No energies parsed from {path}")
        return vals

    def _read_nsp_priors(self, pdbid: str) -> Dict[str, np.ndarray]:
        tsv_path = None
        for cand in (f"{pdbid}.tsv", f"{pdbid}.csv"):
            p = os.path.join(self.paths.nsp_root, cand)
            if os.path.isfile(p):
                tsv_path = p
                break
        if tsv_path is None:
            raise FileNotFoundError(f"NSP3 tsv/csv for {pdbid} not found under {self.paths.nsp_root}")

        df = pd.read_csv(tsv_path, sep=None, engine="python", header=None)
        numeric = df.apply(pd.to_numeric, errors="coerce")
        fin_mask = numeric.notna()
        valid_cols = [i for i in range(numeric.shape[1]) if fin_mask.iloc[:, i].any()]
        if len(valid_cols) < 2:
            raise ValueError("Cannot locate numeric columns for phi/psi in NSP file")
        psi_col = valid_cols[-1]
        phi_col = valid_cols[-2]

        ss8_start = 2
        ss8 = numeric.iloc[:, ss8_start : ss8_start + 8].to_numpy(float)
        ss3_start = ss8_start + 8
        ss3 = numeric.iloc[:, ss3_start : ss3_start + 3].to_numpy(float)

        rsa_col = phi_col - 1
        rsa = numeric.iloc[:, rsa_col].to_numpy(float) if rsa_col >= 0 else None
        phi = numeric.iloc[:, phi_col].to_numpy(float)
        psi = numeric.iloc[:, psi_col].to_numpy(float)

        ss_probs = ss3 if self.ss_mode == "ss3" else ss8
        row_sum = np.clip(ss_probs.sum(axis=1, keepdims=True), 1e-12, None)
        ss_probs = np.where(row_sum > 0, ss_probs / row_sum, ss_probs)

        return {
            "ss_probs": ss_probs,
            "phi": np.deg2rad(phi),
            "psi": np.deg2rad(psi),
            "rsa": None if rsa is None else np.clip(rsa, 0.0, 1.0),
        }

    def _build_candidates(
        self,
        pdbid: str,
        q_dir: str,
        energies: List[float],
        cand_map: Dict[str, str],
    ) -> List[Candidate]:
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
            phi_hat = None
            psi_hat = None
            if angles_df is not None:
                row = angles_df.loc[angles_df["cid"] == cid]
                if len(row) == 1:
                    row = row.iloc[0]
                    phi_cols = [c for c in angles_df.columns if c.startswith("phi_hat_")]
                    psi_cols = [c for c in angles_df.columns if c.startswith("psi_hat_")]
                    phi_hat = np.deg2rad(row[phi_cols].to_numpy(float)) if phi_cols else None
                    psi_hat = np.deg2rad(row[psi_cols].to_numpy(float)) if psi_cols else None

            cands.append(Candidate(
                cid=cid,
                E_q=E_q,
                phi_hat=np.array([]) if phi_hat is None else phi_hat,
                psi_hat=np.array([]) if psi_hat is None else psi_hat,
                ss_probs_hat=None,
                unit="rad",
                meta={"xyz_path": cand_map[cid]},
            ))
        return cands

    def _rerank(self, priors: Dict[str, np.ndarray], cands: List[Candidate]) -> pd.DataFrame:
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
        df_rank.to_csv(os.path.join(out_dir, "rerank.csv"), index=False)
        for new_idx, row in df_rank.reset_index(drop=True).iterrows():
            cid = row["cid"]
            src = cand_map[cid]
            new_name = f"{new_idx+1:02d}_{os.path.basename(src)}"
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
                "If candidate angles are absent, fusion falls back to energy-only ranking.",
                "Provide <pdbid>/angles_<pdbid>.csv with phi_hat_*, psi_hat_* to enable geometric fusion.",
            ],
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":  # optional CLI
    import argparse

    ap = argparse.ArgumentParser(description="E_rerank (fixed)")
    ap.add_argument("--quantum_root", required=True)
    ap.add_argument("--nsp_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--index_file", required=True)
    ap.add_argument("--ss_mode", choices=["ss3","ss8"], default="ss3")
    ap.add_argument("--dist", choices=["l2","kl","ce"], default="ce")
    ap.add_argument("--angle_weight", choices=["uniform","rsa"], default="rsa")
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

