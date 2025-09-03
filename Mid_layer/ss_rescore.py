# --*-- conding:utf-8 --*--
# @time:9/2/25 22:59
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:ss_rescore.py

# Rescoring utilities: read XYZ, compute E_SS terms, combine with E_Q, and re-rank candidates.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from shutil import copy2

from .mid_pip import (
    load_prior,
    compute_dihedrals,
    alpha_hb_scores,
    beta_hb_scores,
    ss_energy_with_terms,
    SSEnergyParams,
    BackboneIndices,
)

# ---------- XYZ backbone parser (expects 'N'/'CA'/'C' atom name token) ----------
def parse_xyz_backbone(xyz_path: Path) -> np.ndarray:
    """
    Parse an .xyz file and extract backbone atoms N, CA, C in residue order.
    Returns a float32 array of shape (L, 3, 3) with order [N, CA, C] per residue.
    Assumes: first two lines may be XYZ header; atom name token is the first column.
    """
    lines = xyz_path.read_text().splitlines()
    start = 2 if len(lines) > 2 and lines[0].strip().isdigit() else 0
    atoms = []
    for ln in lines[start:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        name = parts[0].upper()  # 'N', 'CA', 'C', maybe 'O'
        try:
            x, y, z = float(parts[-3]), float(parts[-2]), float(parts[-1])
        except Exception:
            continue
        atoms.append((name, np.array([x, y, z], dtype=np.float32)))

    # group into residues by encountering triplets (N, CA, C)
    bb = []
    cur = {"N": None, "CA": None, "C": None}
    for name, xyz in atoms:
        if name in ("N", "CA", "C"):
            cur[name] = xyz
            if all(cur[k] is not None for k in ("N", "CA", "C")):
                bb.append(np.stack([cur["N"], cur["CA"], cur["C"]], axis=0))
                cur = {"N": None, "CA": None, "C": None}
    if not bb:
        raise ValueError(f"No N/CA/C triplets in {xyz_path}")
    return np.stack(bb, axis=0)  # (L, 3, 3)

# ---------- Core: compute E_SS for a single XYZ with a fixed prior ----------
def ess_from_xyz(
    prior_path: Path,
    xyz_path: Path,
    params: Optional[SSEnergyParams] = None,
    use_per_residue_mu: bool = True,
) -> float:
    """
    Compute E_SS for a candidate structure (.xyz) under a fixed sequence prior (.npz).
    """
    prior = load_prior(str(prior_path))
    P_ss = prior["P_ss"]
    phi_mu = prior["phi_mu"]
    psi_mu = prior["psi_mu"]
    beta_pairs = prior.get("beta_pairs", np.zeros((0, 3), dtype=np.float32))

    coords = parse_xyz_backbone(xyz_path)
    phi, psi = compute_dihedrals(coords, bb=BackboneIndices(N=0, CA=1, C=2, O=None))
    alpha_list = alpha_hb_scores(coords, P_alpha=P_ss[:, 0])
    beta_list = beta_hb_scores(coords, beta_pairs=beta_pairs)
    mu_tuple = (phi_mu, psi_mu) if use_per_residue_mu else None

    E_SS, _ = ss_energy_with_terms(phi, psi, P_ss, alpha_list, beta_list, params or SSEnergyParams(), use_per_residue_mu=mu_tuple)
    return float(E_SS)

# ---------- Batch rescoring for one protein id ----------
@dataclass
class Candidate:
    tag: str                 # "best", "top_1"... "top_5"
    xyz: Path
    E_Q: Optional[float] = None
    E_SS: Optional[float] = None
    E_total: Optional[float] = None
    terms: Optional[Dict[str, float]] = None  # decomposed E_torsion/E_alphaHB/E_betaHB etc.

@dataclass
class RescoreConfig:
    lam: float = 0.4
    use_per_residue_mu: bool = True
    params: SSEnergyParams = SSEnergyParams()

class Rescorer:
    """
    Post-hoc rescoring for a single protein id directory:
      Result/process_data/best_group/<id>/
        <id>.xyz
        <id>_top_1.xyz ... <id>_top_5.xyz
        top_5_energies_<id>.txt    # optional, one E_Q per line for top_1..top_5
    """
    def __init__(self, id_dir: Path, prior_path: Path, cfg: RescoreConfig = RescoreConfig()):
        self.dir = id_dir
        self.pid = id_dir.name
        self.prior = prior_path
        self.cfg = cfg

    # optional: load E_Q list for top_1..top_5
    def _load_top5_eq(self) -> List[float]:
        path = self.dir / f"top_5_energies_{self.pid}.txt"
        if not path.exists():
            return []
        vals = []
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                vals.append(float(ln))
            except Exception:
                pass
        return vals  # expected length 5

    def gather_candidates(self) -> List[Candidate]:
        cands: List[Candidate] = []
        best_xyz = self.dir / f"{self.pid}.xyz"
        if best_xyz.exists():
            cands.append(Candidate("best", best_xyz))
        for k in range(1, 6):
            p = self.dir / f"{self.pid}_top_{k}.xyz"
            if p.exists():
                cands.append(Candidate(f"top_{k}", p))
        # attach E_Q if available
        eqs = self._load_top5_eq()
        for c in cands:
            if c.tag.startswith("top_"):
                i = int(c.tag.split("_")[1]) - 1
                if i < len(eqs):
                    c.E_Q = float(eqs[i])
        return cands

    def rescore(self) -> List[Candidate]:
        prior = load_prior(str(self.prior))
        P_ss = prior["P_ss"]
        phi_mu = prior["phi_mu"]
        psi_mu = prior["psi_mu"]
        beta_pairs = prior.get("beta_pairs", np.zeros((0, 3), dtype=np.float32))

        cands = self.gather_candidates()
        for c in cands:
            coords = parse_xyz_backbone(c.xyz)
            phi, psi = compute_dihedrals(coords, bb=BackboneIndices(N=0, CA=1, C=2, O=None))
            alpha_list = alpha_hb_scores(coords, P_alpha=P_ss[:, 0])
            beta_list = beta_hb_scores(coords, beta_pairs=beta_pairs)
            mu_tuple = (phi_mu, psi_mu) if self.cfg.use_per_residue_mu else None

            E_SS, terms = ss_energy_with_terms(
                phi=phi,
                psi=psi,
                P_ss=P_ss,
                alpha_scores=alpha_list,
                beta_scores=beta_list,
                params=self.cfg.params,
                use_per_residue_mu=mu_tuple,
            )
            c.E_SS = float(E_SS)
            c.terms = terms
            if c.E_Q is None or not np.isfinite(c.E_Q):
                c.E_total = self.cfg.lam * float(E_SS)
            else:
                c.E_total = float(c.E_Q) + self.cfg.lam * float(E_SS)

        return sorted(cands, key=lambda x: x.E_total)

    def save_ss_best(self, ranked: List[Candidate]) -> Path:
        """Copy the top-ranked candidate by E_total as <id>_ss_best.xyz."""
        ss_best = ranked[0]
        out = self.dir / f"{self.pid}_ss_best.xyz"
        copy2(ss_best.xyz, out)
        return out
