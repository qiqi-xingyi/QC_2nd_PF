# --*-- conding:utf-8 --*--
# @time:8/28/25 22:16
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:mid_pip.py

"""
Hybrid secondary-structure (SS) middleware for AI-augmented VQE
================================================================

Drop-in, standalone Python module that adds an SS-consistency energy term
on top of your existing VQE energy. It **does not** modify your quantum stack.

You provide:
  • quantum_energy_fn(state) -> float                    # your existing VQE energy
  • coords_from_state(state) -> np.ndarray[L, atoms, 3] # how to get backbone coords
  • prior.npz with fields: P_ss[L,3], phi_mu[L], psi_mu[L], beta_pairs[K,3]

We return a scalar objective:
  J(state, step) = E_Q(state) + lambda(step) * E_SS(state)

Quick start (pseudo):
  prior = load_prior("prior.npz")
  wrapper = ObjectiveWrapper(quantum_energy_fn, coords_from_state, prior, default_ss_params())
  J = wrapper.objective(state, step)

All angles are in degrees, distances in Å.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict, Any
import numpy as np

# ---------------------------
# Section 0: Prior I/O
# ---------------------------

def load_prior(path: str) -> Dict[str, np.ndarray]:
    """Load prior.npz with fields:
    - P_ss: (L,3) probabilities in order [alpha, beta, coil]
    - phi_mu, psi_mu: (L,) target angles in degrees
    - beta_pairs: (K,3) each row (p, q, conf), p<q, 0<=conf<=1; may be empty
    """
    data = np.load(path, allow_pickle=False)
    P_ss = data["P_ss"].astype(np.float32)
    phi_mu = data["phi_mu"].astype(np.float32)
    psi_mu = data["psi_mu"].astype(np.float32)
    beta_pairs = data.get("beta_pairs")
    if beta_pairs is None:
        beta_pairs = np.zeros((0,3), dtype=np.float32)
    else:
        beta_pairs = beta_pairs.astype(np.float32)
    return {"P_ss": P_ss, "phi_mu": phi_mu, "psi_mu": psi_mu, "beta_pairs": beta_pairs}

# ---------------------------
# Section 1: Geometry helpers
# ---------------------------

def _torsion(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Return dihedral angle (degrees) for four 3D points.
    Sign follows standard right-hand rule around the bond (p2-p3).
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    # Normal vectors to planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    # Normalize
    def _norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v*0.0
    n1n = _norm(n1)
    n2n = _norm(n2)
    b2n = _norm(b2)
    x = np.dot(n1n, n2n)
    m = np.cross(n1n, b2n)
    y = np.dot(m, n2n)
    ang = np.degrees(np.arctan2(y, x))
    # wrap to [-180, 180]
    if ang > 180: ang -= 360
    if ang <= -180: ang += 360
    return float(ang)

@dataclass
class BackboneIndices:
    N: int
    CA: int
    C: int
    O: int | None = None  # O can be missing; we degrade gracefully

# Default backbone atom order if each residue block is [N, CA, C, O]
DEFAULT_BB = BackboneIndices(N=0, CA=1, C=2, O=3)


def compute_dihedrals(coords: np.ndarray, bb: BackboneIndices = DEFAULT_BB) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Ramachandran dihedrals (phi, psi) from backbone coords.

    Args:
      coords: shape (L, atoms_per_res, 3); requires atoms N, CA, C (O optional)
      bb: indices of backbone atoms inside each residue block

    Returns:
      phi[L], psi[L] in degrees. Endpoints without a well-defined angle are filled with np.nan.
    """
    L = coords.shape[0]
    phi = np.full(L, np.nan, dtype=np.float32)
    psi = np.full(L, np.nan, dtype=np.float32)

    def A(i, idx):
        return coords[i, idx, :]

    # phi(i) = C(i-1) - N(i) - CA(i) - C(i)
    for i in range(1, L):
        C_im1 = A(i-1, bb.C)
        N_i   = A(i,   bb.N)
        CA_i  = A(i,   bb.CA)
        C_i   = A(i,   bb.C)
        phi[i] = _torsion(C_im1, N_i, CA_i, C_i)

    # psi(i) = N(i) - CA(i) - C(i) - N(i+1)
    for i in range(0, L-1):
        N_i   = A(i,   bb.N)
        CA_i  = A(i,   bb.CA)
        C_i   = A(i,   bb.C)
        N_ip1 = A(i+1, bb.N)
        psi[i] = _torsion(N_i, CA_i, C_i, N_ip1)

    return phi, psi

# ---------------------------
# Section 2: H-bond geometry scores
# ---------------------------
@dataclass
class HBKernel:
    d0: float = 2.9      # Å, ideal H-bond distance proxy
    sigma_d: float = 0.25
    theta0: float = 160  # degrees, ideal angle (D-H...A)
    sigma_theta: float = 20


def _gauss(x: float, mu: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((x - mu)/sigma)**2))


def alpha_hb_scores(coords: np.ndarray, P_alpha: np.ndarray, bb: BackboneIndices = DEFAULT_BB, kernel: HBKernel = HBKernel()) -> List[Dict[str, float]]:
    """Compute alpha-helix i→i+4 hydrogen bond geometry scores.

    We use a simple proxy if H is absent: O(i) ... N(i+4) distance and pseudo angle
    formed by C(i)–O(i) ... N(i+4). Returns list of dicts with keys {'i','score'}.
    """
    L = coords.shape[0]
    out: List[Dict[str, float]] = []
    has_O = (DEFAULT_BB.O is not None) and (coords.shape[1] > DEFAULT_BB.O)
    for i in range(0, L-4):
        if P_alpha[i] <= 0 or P_alpha[i+4] <= 0:
            continue
        # donor/acceptor proxy
        if has_O:
            O_i   = coords[i, DEFAULT_BB.O]
            C_i   = coords[i, DEFAULT_BB.C]
        else:
            # fallback: use carbonyl C as proxy for O; weaker signal
            O_i   = coords[i, DEFAULT_BB.C]
            C_i   = coords[i, DEFAULT_BB.C]
        N_ip4 = coords[i+4, DEFAULT_BB.N]
        d = np.linalg.norm(O_i - N_ip4)
        # pseudo angle at acceptor: angle between vector (C->O) and (O->N_{i+4})
        v1 = O_i - C_i
        v2 = N_ip4 - O_i
        def _angle(u,v):
            nu = np.linalg.norm(u); nv = np.linalg.norm(v)
            if nu<1e-8 or nv<1e-8: return 0.0
            c = np.clip(np.dot(u,v)/(nu*nv), -1.0, 1.0)
            return np.degrees(np.arccos(c))
        theta = _angle(v1, v2)
        score = _gauss(d, kernel.d0, kernel.sigma_d) * _gauss(theta, kernel.theta0, kernel.sigma_theta)
        out.append({"i": i, "score": float(score)})
    return out


def beta_hb_scores(coords: np.ndarray, beta_pairs: np.ndarray, bb: BackboneIndices = DEFAULT_BB, kernel: HBKernel = HBKernel()) -> List[Dict[str, float]]:
    """Compute geometry scores for candidate beta pair (p,q) provided by prior.
    We again use O(p) ... N(q) distance + pseudo angle as a smooth score.
    beta_pairs: (K,3) with (p,q,conf)
    Returns list of dicts with keys {'p','q','score'}.
    """
    out: List[Dict[str, float]] = []
    if beta_pairs.size == 0:
        return out
    L = coords.shape[0]
    has_O = (DEFAULT_BB.O is not None) and (coords.shape[1] > DEFAULT_BB.O)
    for (p, q, conf) in beta_pairs:
        p = int(p); q = int(q)
        if p < 0 or q < 0 or p >= L or q >= L: continue
        if has_O:
            O_p = coords[p, DEFAULT_BB.O]
            C_p = coords[p, DEFAULT_BB.C]
        else:
            O_p = coords[p, DEFAULT_BB.C]
            C_p = coords[p, DEFAULT_BB.C]
        N_q = coords[q, DEFAULT_BB.N]
        d = np.linalg.norm(O_p - N_q)
        v1 = O_p - C_p
        v2 = N_q - O_p
        def _angle(u,v):
            nu = np.linalg.norm(u); nv = np.linalg.norm(v)
            if nu<1e-8 or nv<1e-8: return 0.0
            c = np.clip(np.dot(u,v)/(nu*nv), -1.0, 1.0)
            return np.degrees(np.arccos(c))
        theta = _angle(v1, v2)
        geom = _gauss(d, kernel.d0, kernel.sigma_d) * _gauss(theta, kernel.theta0, kernel.sigma_theta)
        score = float(geom) * float(conf)
        out.append({"p": p, "q": q, "score": score})
    return out

# ---------------------------
# Section 3: SS energy
# ---------------------------
@dataclass
class SSEnergyParams:
    # torsion centers (degrees)
    mu_alpha_phi: float = -57.0
    mu_alpha_psi: float = -47.0
    mu_beta_phi:  float = -139.0
    mu_beta_psi:  float = 135.0
    mu_coil_phi:  float = -63.0
    mu_coil_psi:  float = 146.0
    # torsion weights (penalize squared deviation in degrees^2)
    kappa_phi_alpha: float = 0.7
    kappa_psi_alpha: float = 0.7
    kappa_phi_beta:  float = 0.6
    kappa_psi_beta:  float = 0.6
    kappa_phi_coil:  float = 0.1
    kappa_psi_coil:  float = 0.1
    # hydrogen-bond rewards (negative energy contribution)
    kappa_alpha_hb: float = 1.0
    kappa_beta_hb:  float = 1.2


def default_ss_params() -> SSEnergyParams:
    return SSEnergyParams()


def ss_energy(
    phi: np.ndarray,
    psi: np.ndarray,
    P_ss: np.ndarray,  # (L,3) [alpha,beta,coil]
    alpha_scores: List[Dict[str, float]],
    beta_scores:  List[Dict[str, float]],
    params: SSEnergyParams,
    use_per_residue_mu: Tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Compute SS-consistency energy E_SS = E_torsion + E_alphaHB + E_betaHB.

    use_per_residue_mu: optional (phi_mu[L], psi_mu[L]); if given, use per-residue targets
    for torsion centers instead of the class means.
    """
    L = len(phi)
    assert P_ss.shape == (L, 3)

    # --- torsion term ---
    E_tor = 0.0
    for i in range(L):
        if np.isnan(phi[i]) or np.isnan(psi[i]):
            continue
        p_alpha, p_beta, p_coil = float(P_ss[i,0]), float(P_ss[i,1]), float(P_ss[i,2])
        if use_per_residue_mu is None:
            mu_phi_alpha, mu_psi_alpha = params.mu_alpha_phi, params.mu_alpha_psi
            mu_phi_beta,  mu_psi_beta  = params.mu_beta_phi,  params.mu_beta_psi
            mu_phi_coil,  mu_psi_coil  = params.mu_coil_phi,  params.mu_coil_psi
        else:
            # per-residue target from prior prediction (phi_mu, psi_mu)
            mu_phi_alpha = mu_phi_beta = mu_phi_coil = float(use_per_residue_mu[0][i])
            mu_psi_alpha = mu_psi_beta = mu_psi_coil = float(use_per_residue_mu[1][i])
        # squared circular distance in degrees (approx without wrap since we cleaned angles)
        def sq(err):
            return err*err
        E_tor += p_alpha * (
            params.kappa_phi_alpha * sq(phi[i] - mu_phi_alpha)
            + params.kappa_psi_alpha * sq(psi[i] - mu_psi_alpha)
        )
        E_tor += p_beta * (
            params.kappa_phi_beta * sq(phi[i] - mu_phi_beta)
            + params.kappa_psi_beta * sq(psi[i] - mu_psi_beta)
        )
        E_tor += p_coil * (
            params.kappa_phi_coil * sq(phi[i] - mu_phi_coil)
            + params.kappa_psi_coil * sq(psi[i] - mu_psi_coil)
        )

    # --- alpha HB term (reward: negative) ---
    E_a = 0.0
    for g in alpha_scores:
        i = g["i"]
        p = float(P_ss[i,0]) * float(P_ss[i+4,0])  # P_i(alpha)P_{i+4}(alpha)
        E_a += - params.kappa_alpha_hb * p * float(g["score"])

    # --- beta HB term (reward: negative) ---
    E_b = 0.0
    for g in beta_scores:
        p = float(P_ss[g["p"],1]) * float(P_ss[g["q"],1])
        E_b += - params.kappa_beta_hb * p * float(g["score"])

    return float(E_tor + E_a + E_b)

# ---------------------------
# Section 4: Lambda schedule (annealing)
# ---------------------------
@dataclass
class Anneal:
    lambda_max: float = 0.6
    lambda_min: float = 0.3
    warmup_frac: float = 0.3  # fraction of total steps used to decay from max to min

    def value(self, step: int, total_steps: int | None = None) -> float:
        if total_steps is None:
            return self.lambda_min
        cutoff = int(self.warmup_frac * total_steps)
        if step <= cutoff and cutoff > 0:
            # linear
            t = step / max(1, cutoff)
            return float(self.lambda_max + (self.lambda_min - self.lambda_max) * t)
        return float(self.lambda_min)

# ---------------------------
# Section 5: Objective wrapper
# ---------------------------
@dataclass
class ObjectiveWrapper:
    quantum_energy_fn: Callable[[Any], float]
    coords_from_state: Callable[[Any], np.ndarray]
    prior: Dict[str, np.ndarray]
    ss_params: SSEnergyParams = default_ss_params()
    hb_kernel: HBKernel = HBKernel()
    anneal: Anneal = Anneal()
    use_per_residue_mu: bool = False  # if True, use prior's (phi_mu, psi_mu) as torsion centers

    def objective(self, state: Any, step: int, total_steps: int | None = None) -> float:
        """Return J = E_Q + lambda * E_SS for the current state.
        This is the **only** function your classical optimizer needs to call.
        """
        # 1) quantum energy (your black box)
        E_q = float(self.quantum_energy_fn(state))
        # 2) coords and dihedrals
        coords = self.coords_from_state(state)
        phi, psi = compute_dihedrals(coords)
        # 3) H-bond geometry scores
        P_ss = self.prior["P_ss"]
        alpha_list = alpha_hb_scores(coords, P_alpha=P_ss[:,0], kernel=self.hb_kernel)
        beta_list  = beta_hb_scores(coords, beta_pairs=self.prior.get("beta_pairs", np.zeros((0,3), np.float32)), kernel=self.hb_kernel)
        # 4) SS energy
        mu_tuple = None
        if self.use_per_residue_mu:
            mu_tuple = (self.prior["phi_mu"], self.prior["psi_mu"])  # (phi_mu, psi_mu)
        E_ss = ss_energy(phi, psi, P_ss, alpha_list, beta_list, self.ss_params, use_per_residue_mu=mu_tuple)
        # 5) annealed weight
        lam = self.anneal.value(step, total_steps)
        return E_q + lam * E_ss

# ---------------------------
# Section 6: Convenience factory
# ---------------------------

def build_objective(
    quantum_energy_fn: Callable[[Any], float],
    coords_from_state: Callable[[Any], np.ndarray],
    prior_path: str,
    ss_params: SSEnergyParams | None = None,
    hb_kernel: HBKernel | None = None,
    anneal: Anneal | None = None,
    use_per_residue_mu: bool = False,
) -> ObjectiveWrapper:
    prior = load_prior(prior_path)
    return ObjectiveWrapper(
        quantum_energy_fn=quantum_energy_fn,
        coords_from_state=coords_from_state,
        prior=prior,
        ss_params=ss_params or default_ss_params(),
        hb_kernel=hb_kernel or HBKernel(),
        anneal=anneal or Anneal(),
        use_per_residue_mu=use_per_residue_mu,
    )

# ---------------------------
# End of module
# ---------------------------

