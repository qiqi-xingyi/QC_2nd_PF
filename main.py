# --*-- conding:utf-8 --*--
# @time:9/1/25 20:57
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:main.py

"""
Main entry to run E_rerank inside your IDE.

- Uses sensible defaults matching your repo layout (see CONFIG below).
- If `data/data_set/fragments.tsv` is missing, it will try to auto-build an
  index from either `data/seqs_len10/seqs_len10.csv`, `data/seqs/seqs.csv`, or
  the filenames in `nn_result/*.tsv`.
- Prints a short summary and the output location when done.

Run directly (no args needed):
    python main.py

Or override via CLI:
    python main.py \
      --quantum_root data/Quantum_original_data \
      --nsp_root nn_result \
      --out_root E_rerank/out \
      --index_file data/data_set/fragments.tsv \
      --ss_mode ss3 --alpha 1 --beta 1 --gamma 1
"""

from __future__ import annotations
import os
import sys
import logging
from pathlib import Path
import argparse
import pandas as pd

# Ensure local package imports work when running from IDE
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from E_rerank import ERerank
except Exception as e:
    # Helpful error if package not importable
    raise RuntimeError(
        "Cannot import E_rerank. Make sure the folder 'E_rerank' with __init__.py and rerank.py exists "
        "next to this main.py, or install the package into your environment."
    ) from e


# ------------------------------- CONFIG ---------------------------------
DEFAULTS = {
    "quantum_root": PROJECT_ROOT / "data" / "Quantum_original_data",
    "nsp_root": PROJECT_ROOT / "nn_result",
    "out_root": PROJECT_ROOT / "E_rerank" / "out",
    "index_file": PROJECT_ROOT / "data" / "data_set" / "fragments.tsv",
    "ss_mode": "ss3",     # or "ss8"
    "dist": "ce",         # or "kl" / "l2"
    "angle_weight": "rsa", # or "uniform"
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.0,
    "max_candidates": 5,
}


def _autodetect_or_build_index(index_path: Path, nsp_root: Path) -> Path:
    """Return a valid index file path. Build one if missing.

    Priority:
      1) data/data_set/fragments.tsv (expected)
      2) data/seqs_len10/seqs_len10.csv (columns: pdbid[, seq])
      3) data/seqs/seqs.csv (columns: pdbid[, seq])
      4) Infer from nn_result/*.tsv filenames -> runs/_autogen_index.tsv
    """
    if index_path.exists():
        return index_path

    seqs_len10 = PROJECT_ROOT / "data" / "seqs_len10" / "seqs_len10.csv"
    seqs_all = PROJECT_ROOT / "data" / "seqs" / "seqs.csv"
    runs_dir = PROJECT_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    auto_path = runs_dir / "_autogen_index.tsv"

    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        # Accept various header names and normalize to ['pdbid', 'seq']
        lower_map = {c.lower(): c for c in df.columns}
        if "pdbid" in lower_map:
            id_col = lower_map["pdbid"]
        elif "id" in lower_map:
            id_col = lower_map["id"]
        else:
            # fallback: use first column
            id_col = df.columns[0]
        seq_col = lower_map.get("seq")
        out = pd.DataFrame()
        out["pdbid"] = df[id_col].astype(str).str.strip()
        if seq_col is not None:
            out["seq"] = df[seq_col].astype(str)
        return out.dropna(subset=["pdbid"]).drop_duplicates("pdbid")

    # 1) try seqs_len10.csv
    if seqs_len10.exists():
        df = pd.read_csv(seqs_len10)
        _normalize_df(df).to_csv(auto_path, sep="\t", index=False)
        return auto_path

    # 2) try seqs.csv
    if seqs_all.exists():
        df = pd.read_csv(seqs_all)
        _normalize_df(df).to_csv(auto_path, sep="\t", index=False)
        return auto_path

    # 3) fallback from nn_result filenames
    tsvs = sorted([p for p in nsp_root.glob("*.tsv")])
    if not tsvs:
        raise FileNotFoundError(
            "No index file and cannot infer from nn_result/*.tsv. "
            "Please provide data/data_set/fragments.tsv or seqs CSV."
        )
    ids = [p.stem for p in tsvs]
    pd.DataFrame({"pdbid": ids}).to_csv(auto_path, sep="\t", index=False)
    return auto_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run E_rerank from IDE or CLI")
    p.add_argument("--quantum_root", type=str, default=str(DEFAULTS["quantum_root"]))
    p.add_argument("--nsp_root", type=str, default=str(DEFAULTS["nsp_root"]))
    p.add_argument("--out_root", type=str, default=str(DEFAULTS["out_root"]))
    p.add_argument("--index_file", type=str, default=str(DEFAULTS["index_file"]))
    p.add_argument("--ss_mode", choices=["ss3", "ss8"], default=DEFAULTS["ss_mode"])
    p.add_argument("--dist", choices=["ce", "kl", "l2"], default=DEFAULTS["dist"])
    p.add_argument("--angle_weight", choices=["rsa", "uniform"], default=DEFAULTS["angle_weight"])
    p.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    p.add_argument("--beta", type=float, default=DEFAULTS["beta"])
    p.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    p.add_argument("--max_candidates", type=int, default=DEFAULTS["max_candidates"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    log = logging.getLogger("main")

    quantum_root = Path(args.quantum_root).resolve()
    nsp_root = Path(args.nsp_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    index_path = _autodetect_or_build_index(Path(args.index_file), nsp_root)

    log.info("Quantum root : %s", quantum_root)
    log.info("NSP root     : %s", nsp_root)
    log.info("Output root  : %s", out_root)
    log.info("Index file   : %s", index_path)

    rr = ERerank(
        quantum_root=str(quantum_root),
        nsp_root=str(nsp_root),
        out_root=str(out_root),
        ss_mode=args.ss_mode,
        dist=args.dist,
        angle_weight=args.angle_weight,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        max_candidates=args.max_candidates,
    )

    rr.run(index_file=str(index_path))

    log.info("Done. Results are under: %s", out_root)


if __name__ == "__main__":
    main()



