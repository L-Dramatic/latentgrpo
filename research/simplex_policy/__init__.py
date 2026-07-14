"""Exact action-density tools for soft-thought policy optimization."""

from .densities import (
    auxiliary_gumbel_log_density,
    auxiliary_gumbel_score,
    concrete_log_density,
    concrete_score,
    effective_sample_size,
    latent_grpo_surrogate_log_prob,
    ppo_clip_mask,
    sample_concrete,
)

__all__ = [
    "auxiliary_gumbel_log_density",
    "auxiliary_gumbel_score",
    "concrete_log_density",
    "concrete_score",
    "effective_sample_size",
    "latent_grpo_surrogate_log_prob",
    "ppo_clip_mask",
    "sample_concrete",
]
