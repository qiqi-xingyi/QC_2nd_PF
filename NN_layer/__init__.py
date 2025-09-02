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

from .sequence_reader import SequenceReader, Dataset, SeqRecord
from .online_model_client import OnlineModelClient, RawArtifact, RawArtifactIndex
from .prior_postprocessor import PriorPostprocessor, Prior

__all__ = [
    "SequenceReader",
    "Dataset",
    "SeqRecord",
    "OnlineModelClient",
    "RawArtifact",
    "RawArtifactIndex",
    "PriorPostprocessor",
    "Prior",
]
