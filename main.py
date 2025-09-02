# --*-- conding:utf-8 --*--
# @time:9/1/25 20:57
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:main.py

from pathlib import Path
from NN_layer import run_pipeline

def main():
    # user only specifies raw input and final output root
    DATA_DIR = Path("data/seqs")      # input FASTA(s)
    OUT_DIR  = Path("runs")           # root directory for all experiments

    # run full NN_layer pipeline
    priors_dir = run_pipeline(DATA_DIR, OUT_DIR, run_id="exp001")

    print(f"[DONE] Priors generated in {priors_dir}")

if __name__ == "__main__":
    main()


