"""Exact toy contracts for coupled GRPO rollout sampling."""

from .bernoulli_gate import (
    CoupledGroupGate,
    EstimatorMoments,
    JointOutcome,
    antithetic_joint,
    evaluate_coupled_group_gate,
    iid_joint,
)

__all__ = [
    "CoupledGroupGate",
    "EstimatorMoments",
    "JointOutcome",
    "antithetic_joint",
    "evaluate_coupled_group_gate",
    "iid_joint",
]
