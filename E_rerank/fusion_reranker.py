# --*-- conding:utf-8 --*--
# @time:9/7/25 19:43
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:fusion_reranker.py

from __future__ import annotations


from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class Candidate:
    """A single candidate structure."""
    cid: str
    E_q: float
    phi_hat: Optional[np.ndarray] = None
    psi_hat: Optional[np.ndarray] = None
    ss_probs_hat: Optional[np.ndarray] = None
    unit: str = "rad"
    meta: Optional[Dict] = None


class FusionReRanker:
    """
    Combines quantum_rmsd energy and AI priors for re-ranking.

    Parameters
    ----------
    ss_mode : {"ss3","ss8"}
        Which secondary structure distribution to use.
    dist : {"ce","kl","l2"}
        Distance metric for secondary structure distributions.
    angle_weight : {"uniform","rsa"}
        How to weight per-residue angle differences.
    normalize_terms : bool
        Whether to min-max normalize E_q, D_ss, D_phiψ before fusion.
    alpha, beta, gamma : float
        Weights for (E_q, D_ss, D_phiψ).
    """

    def __init__(
        self,
        ss_mode: str = "ss3",
        dist: str = "ce",
        angle_weight: str = "rsa",
        normalize_terms: bool = True,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        self.ss_mode = ss_mode
        self.dist = dist
        self.angle_weight = angle_weight
        self.normalize_terms = normalize_terms
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        # priors (to be loaded later)
        self.ss_probs: Optional[np.ndarray] = None
        self.phi: Optional[np.ndarray] = None
        self.psi: Optional[np.ndarray] = None
        self.rsa: Optional[np.ndarray] = None

    # ------------------- priors -------------------
    def load_priors(
        self,
        ss_probs: np.ndarray,
        phi: np.ndarray,
        psi: np.ndarray,
        rsa: Optional[np.ndarray] = None,
        unit: str = "rad",
    ) -> None:
        """Load NetSurfP priors (secondary structure, angles, rsa)."""
        self.ss_probs = ss_probs
        self.phi = phi
        self.psi = psi
        self.rsa = rsa

        if unit == "deg":
            self.phi = np.deg2rad(self.phi)
            self.psi = np.deg2rad(self.psi)

    # ------------------- scoring -------------------
    def _angle_diff(self, x: np.ndarray, y: np.ndarray) -> float:
        """环形差值 (弧度)，返回均方差."""
        diff = np.angle(np.exp(1j * (x - y)))
        return float(np.mean(diff**2))

    def _ss_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """二级结构分布差异."""
        eps = 1e-9
        if self.dist == "l2":
            return float(np.linalg.norm(p - q))
        elif self.dist == "kl":
            p_ = np.clip(p, eps, 1)
            q_ = np.clip(q, eps, 1)
            return float(np.sum(p_ * np.log(p_ / q_)))
        elif self.dist == "ce":
            q_ = np.clip(q, eps, 1)
            return float(-np.sum(p * np.log(q_)))
        else:
            raise ValueError(f"Unknown dist: {self.dist}")


    def rerank(self, candidates: List[Candidate]) -> pd.DataFrame:
        if self.ss_probs is None or self.phi is None or self.psi is None:
            raise RuntimeError("Must call load_priors before rerank")

        rows = []
        for c in candidates:
            E_q = c.E_q
            D_ss = np.nan
            D_phi_psi = np.nan

            # ---------- D_ss（二级结构分布差异），若候选提供了 ss_probs_hat 才计算 ----------
            if c.ss_probs_hat is not None:
                m = min(len(self.ss_probs), len(c.ss_probs_hat))
                if m > 0:
                    P = self.ss_probs[:m]
                    Q = c.ss_probs_hat[:m]
                    # 只在行向量都有限的残基上计算
                    row_mask = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
                    if np.any(row_mask):
                        idxs = np.where(row_mask)[0]
                        D_ss = float(np.mean([self._ss_distance(P[i], Q[i]) for i in idxs]))

            # ---------- D_{φψ}（角度差），对 NaN 做掩码，只在有效位点上加权 ----------
            if (c.phi_hat is not None and len(c.phi_hat) > 0 and
                    c.psi_hat is not None and len(c.psi_hat) > 0):
                n = min(len(self.phi), len(c.phi_hat), len(self.psi), len(c.psi_hat))
                if n > 0:
                    phi_p = self.phi[:n]
                    psi_p = self.psi[:n]
                    phi_c = c.phi_hat[:n]
                    psi_c = c.psi_hat[:n]

                    mask = (np.isfinite(phi_p) & np.isfinite(psi_p) &
                            np.isfinite(phi_c) & np.isfinite(psi_c))
                    if np.any(mask):
                        # 角度环形差：angle(exp(iΔ)) ∈ [-π, π]
                        dphi = np.angle(np.exp(1j * (phi_p[mask] - phi_c[mask]))) ** 2
                        dpsi = np.angle(np.exp(1j * (psi_p[mask] - psi_c[mask]))) ** 2
                        diff = dphi + dpsi  # 每个有效残基的误差

                        if self.angle_weight == "rsa" and self.rsa is not None:
                            w = np.clip(self.rsa[:n][mask], 0.0, 1.0)
                        else:
                            w = np.ones(diff.shape, dtype=float)

                        # 避免全 0 权重
                        if np.sum(w) <= 1e-12:
                            w = np.ones_like(w, dtype=float)

                        D_phi_psi = float(np.average(diff, weights=w))

            rows.append({
                "cid": c.cid,
                "E_q": E_q,
                "D_ss": D_ss,
                "D_phi_psi": D_phi_psi,
            })

        df = pd.DataFrame(rows)

        # ---------- 列内归一化（忽略 NaN） ----------
        if self.normalize_terms:
            for col in ["E_q", "D_ss", "D_phi_psi"]:
                if col in df.columns:
                    v = df[col].astype(float).to_numpy()
                    finite = np.isfinite(v)
                    if np.any(finite):
                        vmin = np.min(v[finite]);
                        vmax = np.max(v[finite])
                        if vmax > vmin:
                            v_norm = (v - vmin) / (vmax - vmin)
                        else:
                            v_norm = np.zeros_like(v)
                        df[col] = v_norm
                    else:
                        df[col] = v  # 全 NaN 列保持不变

        df["Score"] = (
                self.alpha * df["E_q"].fillna(0.0) +
                self.beta * df["D_ss"].fillna(0.0) +
                self.gamma * df["D_phi_psi"].fillna(0.0)
        )

        return df.sort_values("Score").reset_index(drop=True)

