# --*-- conding:utf-8 --*--
# @time:9/12/25 01:57
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:rmsd_result.py

"""
Summarize RMSD results into one CSV:
Output columns: pdbid, af3, colab, quantum, hybrid
"""

from pathlib import Path
import pandas as pd


AF3_FILE = Path("results/af3_out/af3_rmsd/af3_models.csv")
COLAB_FILE = Path("results/colabfold_out/colabfold_rmsd.csv")
Q_TOP1_FILE = Path("results/quantum_rmsd/quantum_top1.csv")
Q_TOP5_FILE = Path("results/quantum_rmsd/quantum_top5.csv")
OUT_FILE = Path("results/summary_rmsd.csv")


def read_af3(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    agg = (
        df.groupby("pdb_id", as_index=False)["rmsd_rigid_A"]
        .max()
        .rename(columns={"pdb_id": "pdbid", "rmsd_rigid_A": "af3"})
    )
    return agg


def read_colab(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    agg = (
        df.groupby("pdb_id", as_index=False)["rmsd_A"]
        .max()
        .rename(columns={"pdb_id": "pdbid", "rmsd_A": "colab"})
    )
    return agg


def read_quantum_top1(path: Path) -> pd.DataFrame:
    q1 = pd.read_csv(path, header=None)
    rmsd_a = pd.to_numeric(q1.iloc[:, -2], errors="coerce")
    rmsd_b = pd.to_numeric(q1.iloc[:, -3], errors="coerce")
    quantum = pd.DataFrame({
        "pdbid": q1.iloc[:, 0].astype(str),
        "quantum": pd.concat([rmsd_a, rmsd_b], axis=1).min(axis=1)
    })
    return quantum


def read_quantum_top5(path: Path) -> pd.DataFrame:
    q5 = pd.read_csv(path, header=None)
    rmsd_a = pd.to_numeric(q5.iloc[:, -2], errors="coerce")
    rmsd_b = pd.to_numeric(q5.iloc[:, -3], errors="coerce")
    row_min = pd.concat([rmsd_a, rmsd_b], axis=1).min(axis=1)
    tmp = pd.DataFrame({
        "pdbid": q5.iloc[:, 0].astype(str),
        "row_min": row_min
    })
    hybrid = (
        tmp.groupby("pdbid", as_index=False)["row_min"]
        .min()
        .rename(columns={"row_min": "hybrid"})
    )
    return hybrid


def main():
    af3_df = read_af3(AF3_FILE)
    colab_df = read_colab(COLAB_FILE)
    q1_df = read_quantum_top1(Q_TOP1_FILE)
    q5_df = read_quantum_top5(Q_TOP5_FILE)

    out = (
        af3_df.merge(colab_df, on="pdbid", how="outer")
              .merge(q1_df, on="pdbid", how="outer")
              .merge(q5_df, on="pdbid", how="outer")
              .sort_values("pdbid")
              .reset_index(drop=True)
    )

    out = out[["pdbid", "af3", "colab", "quantum", "hybrid"]]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print(f"[OK] Wrote summary to: {OUT_FILE.resolve()}")
    print(f"[INFO] Rows: {len(out)}")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
