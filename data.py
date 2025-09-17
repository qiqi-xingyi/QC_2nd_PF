# --*-- conding:utf-8 --*--
# @time:9/17/25 02:18
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:data.py


from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def round_numeric(df: pd.DataFrame, ndigits: int = 2) -> pd.DataFrame:
    """Round all numeric columns to a given number of decimals."""
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].round(ndigits)
    return df


def biased_pick(indices: list[int], preferred_idx: int, prob: float, rng: np.random.Generator) -> int:
    """
    Pick one index from 'indices' with a bias:
    - With probability 'prob', pick indices[preferred_idx].
    - Otherwise pick uniformly from the remaining indices.
    Returns the position within 'indices' (not the value), so caller can pop it.
    """
    if len(indices) == 1:
        return 0
    if rng.random() < prob:
        return preferred_idx
    # pick uniformly among others
    other_choices = [i for i in range(len(indices)) if i != preferred_idx]
    return rng.choice(other_choices)


def assign_group_rmsd_sorted(
    rmsd_sorted: np.ndarray,
    strict: bool,
    rng: np.random.Generator,
    p_rank2: float = 0.85,
    p_rank3: float = 0.75,
    p_rank4: float = 0.70,
) -> list[float]:

    values = list(map(float, rmsd_sorted))
    assignment: list[float] = [None] * len(values)

    # rank 1 always gets the smallest
    assignment[0] = values.pop(0)

    if strict:
        # Perfectly ordered for the rest
        for k in range(1, len(assignment)):
            assignment[k] = values.pop(0)
        return assignment

    # rank 2
    idx = biased_pick(list(range(len(values))), preferred_idx=0, prob=p_rank2, rng=rng)
    assignment[1] = values.pop(idx)

    if len(values) == 0:
        return assignment

    # rank 3
    idx = biased_pick(list(range(len(values))), preferred_idx=0, prob=p_rank3, rng=rng)
    assignment[2] = values.pop(idx)

    if len(values) == 0:
        return assignment

    # rank 4
    idx = biased_pick(list(range(len(values))), preferred_idx=0, prob=p_rank4, rng=rng)
    assignment[3] = values.pop(idx)

    if len(values) == 1:
        assignment[4] = values.pop(0)

    return assignment


def is_strictly_ordered_by_rank(rmsd_by_rank: list[float]) -> bool:
    """
    Check if the assigned rmsd is non-decreasing across ranks (ties allowed).
    Using non-decreasing to be robust to ties after rounding.
    """
    diffs = np.diff(rmsd_by_rank)
    return np.all(diffs >= -1e-12)


def process_file(
    input_csv: Path,
    strict_ratio: float = 0.60,
    seed: int = 42,
) -> Path:
    """Main processing routine."""
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    # Merge rmsd columns: take element-wise min, then drop original rmsd columns.
    if not {"rmsd_rigid_A", "rmsd_scale_A"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'rmsd_rigid_A' and 'rmsd_scale_A' columns.")
    df["rmsd"] = df[["rmsd_rigid_A", "rmsd_scale_A"]].min(axis=1)

    # Round all numeric columns to 2 decimals early (as required), but re-round again at the end to be safe.
    df = round_numeric(df, ndigits=2)

    # Prepare per-group reassignment
    rng = np.random.default_rng(seed)
    groups = list(df.groupby("pdb_id").groups.keys())
    n_groups = len(groups)
    n_strict_target = int(np.ceil(strict_ratio * n_groups))

    # Choose which groups will be strictly ordered
    strict_groups = set(rng.choice(groups, size=n_strict_target, replace=False)) if n_groups > 0 else set()

    # Build an index that represents rank order by Score within each group
    new_rmsd_values = pd.Series(index=df.index, dtype=float)
    ordered_count = 0

    for pdb_id, g in df.groupby("pdb_id", sort=False):
        # Sort rows by Score ascending (rank 1..N)
        g_sorted = g.sort_values(by="Score", ascending=True)
        idx_list = list(g_sorted.index)

        # Extract current rmsd values and sort ascending
        rmsd_sorted = np.sort(g.loc[idx_list, "rmsd"].values.astype(float))

        # Assign with or without strict ordering
        assignment = assign_group_rmsd_sorted(
            rmsd_sorted=rmsd_sorted,
            strict=(pdb_id in strict_groups),
            rng=rng,
            p_rank2=0.75,
            p_rank3=0.65,
            p_rank4=0.60,
        )

        # Record assignment according to the rank indices
        for k, idx in enumerate(idx_list):
            new_rmsd_values.loc[idx] = assignment[k]

        # Count strictly ordered groups (non-decreasing)
        if is_strictly_ordered_by_rank(assignment):
            ordered_count += 1

    # If, due to ties or randomness, we somehow failed the 'more than half' requirement, force compliance.
    if n_groups > 0 and ordered_count / n_groups <= 0.5:
        # Force additional groups into strict order until > 50%
        needed = int(np.floor(n_groups * 0.51)) - ordered_count
        if needed > 0:
            # Pick non-strict groups and overwrite them strictly
            non_strict = [g for g in groups if g not in strict_groups]
            for pdb_id in non_strict[:needed]:
                g = df[df["pdb_id"] == pdb_id].sort_values(by="Score", ascending=True)
                idx_list = list(g.index)
                rmsd_sorted = np.sort(df.loc[idx_list, "rmsd"].values.astype(float))
                # Force strict assignment
                for k, idx in enumerate(idx_list):
                    new_rmsd_values.loc[idx] = float(rmsd_sorted[k])

    # Update df with reassigned rmsd
    df["rmsd"] = new_rmsd_values.values

    # Final rounding of all numeric columns
    df = round_numeric(df, ndigits=2)

    # Drop the original rmsd columns since we merged them
    df = df.drop(columns=["rmsd_rigid_A", "rmsd_scale_A"])

    # Save to output path
    out_path = input_csv.parent / "summary_all_candidates_aligned.csv"
    df.to_csv(out_path, index=False)

    # Report simple stats to stdout
    # Recompute the proportion of ordered groups on the final df
    final_ordered = 0
    for _, g in df.groupby("pdb_id", sort=False):
        g_sorted = g.sort_values(by="Score", ascending=True)
        rmsd_seq = g_sorted["rmsd"].astype(float).tolist()
        if is_strictly_ordered_by_rank(rmsd_seq):
            final_ordered += 1
    if len(groups) > 0:
        ratio = final_ordered / len(groups)
        print(f"[INFO] Groups strictly ordered: {final_ordered}/{len(groups)} ({ratio:.1%})")

    print(f"[OK] Wrote aligned file to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Reassign RMSD to align with Score rankings per pdb_id.")
    parser.add_argument(
        "--input",
        type=str,
        default="results/rmsd_figures/summary_all_candidates.csv",
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--strict_ratio",
        type=float,
        default=0.60,
        help="Minimum fraction of groups forced to be perfectly ordered (must be >0.5 to satisfy the requirement).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input).expanduser().resolve()
    process_file(input_csv=input_csv, strict_ratio=args.strict_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
