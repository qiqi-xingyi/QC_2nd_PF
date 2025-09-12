# --*-- conding:utf-8 --*--
# @time:8/28/25 23:28
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:__init__.py


"""E_rerank package.

Example:
    from E_rerank import ERerank, run

    run(
        quantum_root="./data/Quantum_original_data",
        nsp_root="./nn_result",
        out_root="./E_rerank/out",
        index_file="./data/data_set/fragments.tsv",
        ss_mode="ss3",
        alpha=1, beta=1, gamma=1,
    )
"""

from .rerank import ERerank  # re-export

__all__ = ["ERerank", "run"]


def run(
    *,
    quantum_root: str,
    nsp_root: str,
    out_root: str,
    index_file: str,
    ss_mode: str = "ss3",
    dist: str = "ce",
    angle_weight: str = "rsa",
    normalize_terms: bool = True,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    max_candidates: int = 5,
) -> None:
    """Convenience runner for scripting usage.

    This function simply constructs :class:`ERerank` and calls ``run``.
    """
    rr = ERerank(
        quantum_root=quantum_root,
        nsp_root=nsp_root,
        out_root=out_root,
        ss_mode=ss_mode,
        dist=dist,
        angle_weight=angle_weight,
        normalize_terms=normalize_terms,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_candidates=max_candidates,
    )
    rr.run(index_file=index_file)


# Optional: allow `python -m E_rerank` to behave like a CLI
if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="E_rerank: fuse NSP priors and quantum_rmsd energies to re-rank candidates")
    ap.add_argument("--quantum_root", required=True)
    ap.add_argument("--nsp_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--index_file", required=True, help="CSV/TSV with column 'pdbid'")
    ap.add_argument("--ss_mode", choices=["ss3","ss8"], default="ss3")
    ap.add_argument("--dist", choices=["l2","kl","ce"], default="ce")
    ap.add_argument("--angle_weight", choices=["uniform","rsa"], default="rsa")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--max_candidates", type=int, default=5)

    args = ap.parse_args()
    run(
        quantum_root=args.quantum_root,
        nsp_root=args.nsp_root,
        out_root=args.out_root,
        index_file=args.index_file,
        ss_mode=args.ss_mode,
        dist=args.dist,
        angle_weight=args.angle_weight,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        max_candidates=args.max_candidates,
    )
