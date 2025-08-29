# --*-- conding:utf-8 --*--
# @time:8/29/25 01:34
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:prior_postprocessor.py

# NN_layer/prior_postprocessor.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
import time
import hashlib

import numpy as np
import pandas as pd

# If you placed these in another module, adjust imports accordingly:
from NN_layer.online_model_client import RawArtifact, RawArtifactIndex


# ---------------------------
# Unified Prior data object
# ---------------------------

@dataclass
class Prior:
    """Unified prior for the Mid layer (secondary-structure guidance)."""
    P_ss: np.ndarray        # (L, 3) float32 -> columns [alpha, beta, coil]
    phi_mu: np.ndarray      # (L,) float32, degrees in [-180, 180]
    psi_mu: np.ndarray      # (L,) float32, degrees in [-180, 180]
    beta_pairs: np.ndarray  # (K, 3) float32: (p, q, conf). Can be empty.
    meta: Dict              # provenance and preprocessing metadata


# ---------------------------
# Utilities
# ---------------------------

_Q8_ORDER = ["H", "E", "C", "G", "I", "B", "T", "S"]
_Q8_TO_Q3 = {
    "H": "alpha", "G": "alpha", "I": "alpha",
    "E": "beta",  "B": "beta",
    "C": "coil",  "T": "coil",  "S": "coil"
}
_Q3_INDEX = {"alpha": 0, "beta": 1, "coil": 2}

_TYPICAL_ANGLES = {
    "alpha": {"phi": -57.0,  "psi": -47.0},
    "beta":  {"phi": -139.0, "psi": 135.0},
    "coil":  {"phi": -63.0,  "psi": 146.0},
}


def _wrap_deg(a: np.ndarray) -> np.ndarray:
    """Wrap degrees to [-180, 180]."""
    return ((a + 180.0) % 360.0) - 180.0


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _safe_name(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


# ---------------------------
# Postprocessor class
# ---------------------------

class PriorPostprocessor:
    """
    Parse NetSurfP-3 raw tables (TSV/CSV) and convert them to a unified `Prior`:
      - Q8 probabilities/labels -> Q3 probabilities (alpha, beta, coil)
      - phi/psi angles (degrees) -> wrapped to [-180, 180]
      - beta_pairs left empty for now (can be filled by other modules)

    Typical usage:
        pp = PriorPostprocessor()
        prior = pp.parse_one(artifact)
        pp.save(prior, out_npz, out_meta)
        # Or batch:
        paths = pp.batch(raw_index, out_dir)
    """

    def __init__(
        self,
        q8_prob_pattern: str = r"(?i)^(Q8_)?[HECGIBTS]$",
        label_candidates: Tuple[str, ...] = ("Q8", "SS8", "ss8", "label", "Label"),
        phi_candidates: Tuple[str, ...] = ("phi", "PHI", "phi_pred", "Phi"),
        psi_candidates: Tuple[str, ...] = ("psi", "PSI", "psi_pred", "Psi"),
        typical_angles: Dict[str, Dict[str, float]] = None,
        min_prob_clip: float = 1e-6,  # to avoid division by zero when normalizing
        verbose: bool = True,
    ):
        self.q8_prob_pattern = re.compile(q8_prob_pattern)
        self.label_candidates = label_candidates
        self.phi_candidates = phi_candidates
        self.psi_candidates = psi_candidates
        self.typical_angles = typical_angles or _TYPICAL_ANGLES
        self.min_prob_clip = float(min_prob_clip)
        self.verbose = verbose

    # --------- Public API ---------

    def parse_one(self, artifact: RawArtifact) -> Prior:
        """
        Convert one RawArtifact's main table (TSV/CSV) into a Prior.

        Steps:
          1) Read table (auto delimiter).
          2) Extract Q8 probabilities (preferred) or labels -> P_q8.
          3) Aggregate Q8 -> Q3 probabilities, normalize row-wise.
          4) Extract phi/psi columns (degrees), wrapped to [-180, 180].
             If missing or invalid, fall back to typical angles per residue
             using the argmax Q3 class (alpha/beta/coil).
          5) Create Prior (beta_pairs empty).
        """
        if artifact.main_table is None or not Path(artifact.main_table).exists():
            raise FileNotFoundError(
                f"No main table (TSV/CSV) found for seq_id={artifact.seq_id} "
                f"in {artifact.output_dir}"
            )

        df = pd.read_csv(artifact.main_table, sep=None, engine="python")

        # 2) Q8 probs or labels
        P_q8 = self._extract_q8_probs_or_onehot(df)

        # 3) Q8 -> Q3
        P_ss = self._q8_to_q3_probs(P_q8)  # (L, 3)

        # 4) phi/psi (degrees)
        phi, psi = self._extract_phi_psi(df, P_ss)

        # 5) assemble Prior
        L = P_ss.shape[0]
        beta_pairs = np.zeros((0, 3), dtype=np.float32)  # empty for now
        meta = {
            "provider": "netsurfp_biolib",
            "parser": "PriorPostprocessor",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seq_id": artifact.seq_id,
            "input_fasta": str(artifact.input_fasta),
            "source_table": str(artifact.main_table),
            "length": int(L),
            "notes": "Q8→Q3 aggregation; angles wrapped; missing angles fallback to typical centers",
        }
        return Prior(
            P_ss=P_ss.astype(np.float32),
            phi_mu=phi.astype(np.float32),
            psi_mu=psi.astype(np.float32),
            beta_pairs=beta_pairs,
            meta=meta,
        )

    def save(self, prior: Prior, out_npz: Path, out_meta: Path) -> None:
        """Persist prior as .npz and a sidecar meta.json."""
        out_npz = Path(out_npz)
        out_meta = Path(out_meta)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        out_meta.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            out_npz,
            P_ss=prior.P_ss,
            phi_mu=prior.phi_mu,
            psi_mu=prior.psi_mu,
            beta_pairs=prior.beta_pairs,
        )
        out_meta.write_text(json.dumps(prior.meta, ensure_ascii=False, indent=2))

    def batch(self, raw_index: RawArtifactIndex, out_dir: Path) -> List[Path]:
        """
        Convert a collection of RawArtifacts into multiple prior files.
        Returns the list of produced .npz paths.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        produced: List[Path] = []
        for i, art in enumerate(raw_index.items):
            try:
                if self.verbose:
                    print(f"[{i+1}/{len(raw_index.items)}] Parsing seq_id={art.seq_id}")
                prior = self.parse_one(art)

                # Name prior files by seq_id + length + short hash of input_fasta
                base = f"{_safe_name(art.seq_id)}_L{prior.P_ss.shape[0]}_{_sha1_text(str(art.input_fasta))[:8]}"
                out_npz = out_dir / f"{base}.prior.npz"
                out_meta = out_dir / f"{base}.meta.json"

                # augment meta with run-level info if available
                meta_aug = dict(prior.meta)
                meta_aug["run_meta"] = raw_index.meta
                prior.meta = meta_aug

                self.save(prior, out_npz, out_meta)
                produced.append(out_npz)
            except Exception as e:
                if self.verbose:
                    print(f"  -> parse failed for {art.seq_id}: {e}")
        # Write a small summary index
        (out_dir / "prior_index.json").write_text(
            json.dumps(
                {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "count": len(produced),
                    "paths": [str(p) for p in produced],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if self.verbose:
            print(f"[OK] Prior generation: {len(produced)} files -> {out_dir}")
        return produced

    # --------- Internals ---------

    def _extract_q8_probs_or_onehot(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return P_q8 (L, 8) as float32. Prefer probability columns; otherwise
        derive one-hot from a Q8 label column.
        """
        # Find Q8 prob columns by regex
        prob_cols = [c for c in df.columns if self.q8_prob_pattern.match(str(c))]
        has_probs = len(prob_cols) >= 8

        if has_probs:
            # Build column order matching _Q8_ORDER
            ordered_cols: List[str] = []
            for sym in _Q8_ORDER:
                # accept exact or suffix match (e.g., "Q8_H" or "H")
                matches = [c for c in prob_cols if str(c).upper().endswith(sym)]
                if not matches:
                    raise KeyError(f"Missing Q8 probability column for {sym}")
                ordered_cols.append(matches[0])
            P_q8 = df[ordered_cols].to_numpy(dtype=np.float32)
            # Normalize per row
            row_sum = P_q8.sum(axis=1, keepdims=True)
            row_sum = np.clip(row_sum, self.min_prob_clip, None)
            P_q8 = (P_q8 / row_sum).astype(np.float32)
            return P_q8

        # Otherwise, look for a label column to build one-hot
        lab_col = None
        for cand in self.label_candidates:
            if cand in df.columns:
                lab_col = cand
                break
        if lab_col is None:
            raise KeyError(
                "Could not find Q8 probability columns or a Q8 label column "
                f"in table columns: {list(df.columns)[:20]} ..."
            )
        labels = (
            df[lab_col].astype(str).str.strip().str.upper().str[0].tolist()
        )
        L = len(labels)
        idx = {s: j for j, s in enumerate(_Q8_ORDER)}
        P_q8 = np.zeros((L, len(_Q8_ORDER)), dtype=np.float32)
        for i, s in enumerate(labels):
            j = idx.get(s, idx["C"])  # default to 'C' if unknown
            P_q8[i, j] = 1.0
        return P_q8

    def _q8_to_q3_probs(self, P_q8: np.ndarray) -> np.ndarray:
        """
        Aggregate Q8 probabilities (H,E,C,G,I,B,T,S) into Q3 (alpha,beta,coil).
        """
        L = P_q8.shape[0]
        P_ss = np.zeros((L, 3), dtype=np.float32)
        for j, sym in enumerate(_Q8_ORDER):
            q3 = _Q8_TO_Q3.get(sym, "coil")
            P_ss[:, _Q3_INDEX[q3]] += P_q8[:, j]
        # normalize row-wise (defensive)
        row_sum = P_ss.sum(axis=1, keepdims=True)
        row_sum = np.clip(row_sum, self.min_prob_clip, None)
        P_ss = (P_ss / row_sum).astype(np.float32)
        return P_ss

    def _extract_phi_psi(self, df: pd.DataFrame, P_ss: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract phi/psi (degrees). If columns are missing or contain NaNs,
        fallback to typical angles based on argmax Q3 class per residue.
        """
        def _pick(colnames: Tuple[str, ...]) -> Optional[str]:
            for n in colnames:
                if n in df.columns:
                    return n
            return None

        phi_col = _pick(self.phi_candidates)
        psi_col = _pick(self.psi_candidates)

        L = P_ss.shape[0]
        phi = np.full(L, np.nan, dtype=np.float32)
        psi = np.full(L, np.nan, dtype=np.float32)

        if phi_col is not None:
            phi = pd.to_numeric(df[phi_col], errors="coerce").to_numpy(dtype=np.float32)
        if psi_col is not None:
            psi = pd.to_numeric(df[psi_col], errors="coerce").to_numpy(dtype=np.float32)

        # Wrap to [-180, 180]
        if np.any(np.isfinite(phi)):
            phi = _wrap_deg(phi)
        if np.any(np.isfinite(psi)):
            psi = _wrap_deg(psi)

        # Fallback for missing/NaN: typical angle of the argmax Q3 class
        bad_phi = ~np.isfinite(phi)
        bad_psi = ~np.isfinite(psi)
        if np.any(bad_phi) or np.any(bad_psi):
            # choose class per residue
            cls_idx = np.argmax(P_ss, axis=1)  # 0=alpha,1=beta,2=coil
            cls_name = np.array(["alpha", "beta", "coil"])[cls_idx]
            # Prepare arrays of typical angles
            typ_phi = np.array([self.typical_angles[c]["phi"] for c in cls_name], dtype=np.float32)
            typ_psi = np.array([self.typical_angles[c]["psi"] for c in cls_name], dtype=np.float32)
            phi[bad_phi] = typ_phi[bad_phi]
            psi[bad_psi] = typ_psi[bad_psi]

        return phi.astype(np.float32), psi.astype(np.float32)
