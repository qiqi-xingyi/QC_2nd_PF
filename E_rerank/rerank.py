# --*-- conding:utf-8 --*--
# @time:9/7/25 19:14
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:rerank.py

# ─────────────────────────────────────────────────────────────────────────────
# File: E_rerank/rerank.py
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
"""
E_rerank.rerank

Core class to fuse NetSurfP-3.0 priors with quantum energies and re-rank
candidate structures. Also provides a small CLI for batch processing.

Typical usage (library):

    from E_rerank import ERerank
    rr = ERerank(
        quantum_root="./data/Quantum_original_data",
        nsp_root="./nn_result",
        out_root="./E_rerank/out",
        ss_mode="ss3",
        alpha=1.0, beta=1.0, gamma=1.0,
    )
    rr.run(index_file="./data/data_set/fragments.tsv")

CLI (after installing the package):

    python -m E_rerank \
      --quantum_root ./data/Quantum_original_data \
      --nsp_root ./nn_result \
      --out_root ./E_rerank/out \
      --index_file ./data/data_set/fragments.tsv \
      --ss_mode ss3 --alpha 1 --beta 1 --gamma 1

Directory expectation (per pdbid under quantum_root):
    <pdbid>/
        <pdbid>_top_1.xyz .. <pdbid>_top_5.xyz
        top_5_energies_<pdbid>.txt | energy_list_<pdbid>.txt | energies_<pdbid>.txt
    Optional: angles_<pdbid>.csv with columns
        cid, phi_hat_0..phi_hat_{L-1}, psi_hat_0..psi_hat_{L-1}

NetSurfP TSV (under nsp_root):
    <pdbid>.tsv (or .csv) with columns like:
        idx, aa, ss8_p0..ss8_p7, ss3_p0..ss3_p2, dis_p0, dis_p1, rsa, asa, phi, psi
    (Headerless files supported; columns detected heuristically.)
"""

import os
import json
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Local import (ensure fusion_reranker.py is installed/available on path)
try:
    from fusion_reranker import FusionReRanker, Candidate
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
    """Fuse NetSurfP priors with quantum energies and re-rank candidates.

    Parameters
    ----------
    quantum_root : str
        Root folder containing per-pdbid subfolders with xyz and energy files.
    nsp_root : str
        Folder containing NetSurfP outputs with filenames <pdbid>.tsv or .csv.
    out_root : str
        Output root. Per-pdbid results will be created here.
    ss_mode : {"ss3","ss8"}
        Which prior distribution to use (3-class or 8-class secondary structure).
    dist : {"l2","kl","ce"}
        Distance/divergence for SS distribution mismatch.
    angle_weight : {"uniform","rsa"}
        Residue-wise angle weights. If "rsa", uses NetSurfP RSA when available.
    normalize_terms : bool
        Whether to min–max normalize E_q, D_ss, D_phi_psi before fusion.
    alpha, beta, gamma : float
        Fusion weights for (E_q, D_ss, D_phi_psi).
    max_candidates : int
        Number of top-k xyz candidates per pdbid to consider.
    random_seed : int
        RNG seed for any stochastic fallback (currently unused).
    """

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

    # ---------------------------- public API ----------------------------
    def run(self, index_file: str) -> None:
        """Process all PDBIDs listed in index_file.

        index_file must be CSV/TSV with at least a column 'pdbid'.
        (Additional columns, e.g., 'seq', are accepted and ignored.)
        """
        idx = pd.read_csv(index_file, sep=None, engine="python")
        if "pdbid" not in idx.columns:
            raise ValueError("index_file must contain column 'pdbid'")
        for pdbid in idx["pdbid"].astype(str):
            self._process_one(pdbid)

    # ---------------------------- core steps ----------------------------
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

        # discover xyz files
        cand_map: Dict[str, str] = {}
        for i in range(1, 100):
            p = os.path.join(q_dir, f"{pdbid}_top_{i}.xyz")
            if os.path.isfile(p):
                cand_map[f"top_{i}"] = p
            else:
                break
        if not cand_map:
            raise FileNotFoundError(f"No candidate xyz files like {pdbid}_top_1.xyz under {q_dir}")
        # limit to max_candidates
        cand_map = dict(list(sorted(cand_map.items(), key=lambda x: int(x[0].split('_')[-1])))[: self.max_candidates])

        # read energies
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
            # generic csv/tsv with 'energy' column
            for fn in os.listdir(q_dir):
                if fn.lower().endswith((".csv", ".tsv")) and "energy" in fn.lower():
                    energies = self._read_energy_file(os.path.join(q_dir, fn))
                    break
        if energies is None:
            raise FileNotFoundError(f"Cannot find an energy file in {q_dir}")

        # align length
        if len(energies) < len(cand_map):
            raise ValueError(f"Energy list shorter ({len(energies)}) than candidates ({len(cand_map)})")
        energies = energies[: len(cand_map)]
        return energies, cand_map

    def _read_energy_file(self, path: str) -> List[float]:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".tsv", ".txt"):
            try:
                df = pd.read_csv(path, sep=None, engine="python")
                for col in df.columns:
                    if col.lower() in {"energy", "energies", "e", "ene", "value"}:
                        return df[col].astype(float).tolist()
            except Exception:
                pass
        # fallback: one value per line
        vals: List[float] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().split()
                if not s:
                    continue
                try:
                    vals.append(float(s[0]))
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
        # locate phi/psi as last two numeric columns
        fin_mask = numeric.notna()
        valid_cols = [i for i in range(numeric.shape[1]) if fin_mask.iloc[:, i].any()]
        if len(valid_cols) < 2:
            raise ValueError("Cannot locate numeric columns for phi/psi in NSP file")
        psi_col = valid_cols[-1]
        phi_col = valid_cols[-2]

        # ss8 block is 8 columns right after aa (index=2..9), ss3 block is next 3 columns
        ss8_start = 2
        ss8 = numeric.iloc[:, ss8_start : ss8_start + 8].to_numpy(float)
        ss3_start = ss8_start + 8
        ss3 = numeric.iloc[:, ss3_start : ss3_start + 3].to_numpy(float)

        rsa_col = phi_col - 1
        rsa = numeric.iloc[:, rsa_col].to_numpy(float) if rsa_col >= 0 else None
        phi = numeric.iloc[:, phi_col].to_numpy(float)
        psi = numeric.iloc[:, psi_col].to_numpy(float)

        ss_probs = ss3 if self.ss_mode == "ss3" else ss8
        # row-normalize
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
