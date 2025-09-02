# --*-- conding:utf-8 --*--
# @time:8/28/25 23:28
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:__init__.py.py

"""
NN_layer package

This package provides the neural-network prior interface for the hybrid AI–Quantum workflow.
It contains three main modules:

- sequence_reader: Read and standardize protein sequences from FASTA, produce Dataset.
- online_model_client: Call NetSurfP-3.0 via BioLib to generate raw predictions.
- prior_postprocessor: Convert raw predictions into a unified Prior object (P_ss, φ/ψ, beta_pairs).

Typical usage:
    from NN_layer import SequenceReader, OnlineModelClient, PriorPostprocessor
"""

from pathlib import Path
import time
from .sequence_reader import SequenceReader
from .online_model_client import OnlineModelClient
from .prior_postprocessor import PriorPostprocessor

def run_pipeline(data_path: str | Path, out_dir: str | Path, run_id: str | None = None) -> Path:
    """
    High-level entry point: run full NN_layer pipeline.
    Args:
        data_path: Path to raw FASTA file or directory
        out_dir: Root directory for outputs
        run_id: Optional run name (default: timestamp)
    Returns:
        Path to priors directory containing final prior .npz files
    """
    out_dir = Path(out_dir)
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_root = out_dir / run_id
    inputs_dir = run_root / "inputs"
    raw_dir = run_root / "raw"
    priors_dir = run_root / "priors"

    # Step 1
    reader = SequenceReader(window_size=None, stride=1, dedup=True, min_len=3)
    pairs = reader.load(data_path)
    dataset = reader.build_dataset(pairs)
    dataset = reader.save_fasta(dataset, inputs_dir)

    # Step 2
    client = OnlineModelClient(app_id="DTU/NetSurfP-3", retries=1, verbose=True)
    raw_index = client.predict(dataset, raw_dir)

    # Step 3
    pp = PriorPostprocessor()
    pp.batch(raw_index, priors_dir)

    return priors_dir

