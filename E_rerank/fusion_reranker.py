# --*-- conding:utf-8 --*--
# @time:9/7/25 19:43
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:fusion_reranker.py

from __future__ import annotations
"""
fusion_reranker.py

Implements Candidate dataclass and FusionReRanker class.

- Candidate: 包装单个候选结构的信息（量子能量、角度、SS 概率等）。
- FusionReRanker: 结合 NetSurfP 预测的先验 (SS, φ/ψ, RSA) 与量子能量 E_q，
  计算融合打分并重排候选结构。

Author: Yuqi Zhang (with ChatGPT)
"""

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
    Combines quantum energy and AI priors for re-ranking.

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

            # secondary structure distance (如果候选没有提供概率，跳过)
            if c.ss_probs_hat is not None:
                n = min(len(self.ss_probs), len(c.ss_probs_hat))
                D_ss = np.mean(
                    [self._ss_distance(self.ss_probs[i], c.ss_probs_hat[i]) for i in range(n)]
                )

            # angle difference (如果候选提供了 phi/psi)
            if c.phi_hat is not None and len(c.phi_hat) > 0:
                n = min(len(self.phi), len(c.phi_hat))
                w = np.ones(n)
                if self.angle_weight == "rsa" and self.rsa is not None:
                    w = self.rsa[:n]
                D_phi = [self._angle_diff(self.phi[i:i+1], c.phi_hat[i:i+1]) for i in range(n)]
                D_psi = [self._angle_diff(self.psi[i:i+1], c.psi_hat[i:i+1]) for i in range(n)]
                D_phi_psi = float(np.average(np.array(D_phi) + np.array(D_psi), weights=w))

            rows.append({
                "cid": c.cid,
                "E_q": E_q,
                "D_ss": D_ss,
                "D_phi_psi": D_phi_psi,
            })

        df = pd.DataFrame(rows)

        # normalization
        if self.normalize_terms:
            for col in ["E_q", "D_ss", "D_phi_psi"]:
                if df[col].notna().any():
                    v = df[col].astype(float).to_numpy()
                    vmin, vmax = np.nanmin(v), np.nanmax(v)
                    if vmax > vmin:
                        df[col] = (v - vmin) / (vmax - vmin)
                    else:
                        df[col] = 0.0

        df["Score"] = (
            self.alpha * df["E_q"].fillna(0)
            + self.beta * df["D_ss"].fillna(0)
            + self.gamma * df["D_phi_psi"].fillna(0)
        )

        return df.sort_values("Score").reset_index(drop=True)
