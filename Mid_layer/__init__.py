# --*-- conding:utf-8 --*--
# @time:8/28/25 23:28
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:__init__.py

"""
Mid_layer package

This package provides middleware utilities to combine AI-derived secondary-structure
priors with a quantum VQE workflow. It supports both:
  - Post-hoc rescoring: E_total = E_Q + lambda * E_SS (after VQE produced candidates)
  - In-loop guidance:  J(state,t) = E_Q(state) + lambda(t) * E_SS(state)

Public API:
  - load_prior, compute_dihedrals, ss_energy, ss_energy_with_terms,
    default_ss_params, SSEnergyParams, HBKernel, Anneal, BackboneIndices,
    ObjectiveWrapper, build_objective
  - parse_xyz_backbone, ess_from_xyz, Rescorer, RescoreConfig
"""

from .mid_pip import (
    load_prior,
    compute_dihedrals,
    ss_energy,
    ss_energy_with_terms,
    default_ss_params,
    SSEnergyParams,
    HBKernel,
    Anneal,
    BackboneIndices,
    ObjectiveWrapper,
    build_objective,
)

from .ss_rescore import (
    parse_xyz_backbone,
    ess_from_xyz,
    Rescorer,
    RescoreConfig,
)

__all__ = [
    # core
    "load_prior",
    "compute_dihedrals",
    "ss_energy",
    "ss_energy_with_terms",
    "default_ss_params",
    "SSEnergyParams",
    "HBKernel",
    "Anneal",
    "BackboneIndices",
    "ObjectiveWrapper",
    "build_objective",
    # rescoring tools
    "parse_xyz_backbone",
    "ess_from_xyz",
    "Rescorer",
    "RescoreConfig",
]
