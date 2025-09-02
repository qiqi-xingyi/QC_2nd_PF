# --*-- conding:utf-8 --*--
# @time:9/1/25 20:57
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:main.py

from pathlib import Path
from NN_layer import SequenceReader, OnlineModelClient, PriorPostprocessor

def main():
    # === Configure external paths ===
    DATA_DIR   = Path("data/seqs")        # directory with raw FASTA sequences
    RUN_DIR    = Path("runs/exp001")      # root directory for this experiment
    INPUTS_DIR = RUN_DIR / "inputs"       # standardized FASTA inputs
    RAW_DIR    = RUN_DIR / "raw"          # raw outputs from NetSurfP-3
    PRIORS_DIR = RUN_DIR / "priors"       # processed priors for downstream use

    # Step 1: Read and standardize input sequences
    reader = SequenceReader(window_size=None, stride=1, dedup=True, min_len=3)
    pairs = reader.load(DATA_DIR)                     # read FASTA(s)
    dataset = reader.build_dataset(pairs)
    dataset = reader.save_fasta(dataset, INPUTS_DIR)  # write standardized FASTA + index.json

    # Step 2: Run online prediction using NetSurfP-3 (via BioLib)
    client = OnlineModelClient(
        app_id="DTU/NetSurfP-3",
        retries=1,
        rate_limit_s=0.0,    # increase if you need to throttle requests
        verbose=True,
    )
    raw_index = client.predict(dataset, RAW_DIR)      # raw_index.json + per-sequence outputs

    # Step 3: Post-process raw results into unified Prior format
    pp = PriorPostprocessor()
    prior_paths = pp.batch(raw_index, PRIORS_DIR)     # *.prior.npz + *.meta.json + prior_index.json

    print(f"[DONE] Generated {len(prior_paths)} priors under {PRIORS_DIR}")

if __name__ == "__main__":
    main()

