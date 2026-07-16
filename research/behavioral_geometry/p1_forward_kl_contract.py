"""Pure contracts for endpoint-aware finite-horizon continuation KL.

This module intentionally has no checkpoint, SGLang, dataset, or GPU dependency.
It fixes the probability-space semantics before the real source-action preflight
is allowed to inspect model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class ContinuationEndpoint(str, Enum):
    """Mutually exclusive latent-closure endpoints used by the P1 redesign."""

    NATURAL_VISIBLE = "NATURAL_VISIBLE"
    FINISH_LENGTH = "LATENT_FINISH_LENGTH"
    FINISH_TOKEN = "LATENT_FINISH_TOKEN"
    FINISH_STRING = "LATENT_FINISH_STRING"
    LATENT_TIMEOUT = "LATENT_TIMEOUT"
    EXECUTION_ABORT = "EXECUTION_ABORT"
    FORCED_VISIBLE_CONTROL = "FORCED_VISIBLE_CONTROL"


@dataclass(frozen=True)
class EndpointPairAudit:
    reference: ContinuationEndpoint
    candidate: ContinuationEndpoint
    same_atom: bool
    endpoint_kl: float
    visible_continuation_defined: bool
    scientific_visible_pair: bool


@dataclass(frozen=True)
class PairedPathKL:
    """Conditional KL values evaluated on one policy's sampled histories."""

    per_step: Tensor
    total: float


def audit_endpoint_pair(
    reference: ContinuationEndpoint,
    candidate: ContinuationEndpoint,
    *,
    allow_forced_control: bool = False,
) -> EndpointPairAudit:
    """Audit two deterministic closure endpoints on a common atom space.

    A forced boundary is never a scientific endpoint. It may be admitted only
    to exercise visible sample/teacher-force plumbing.
    """

    if not isinstance(reference, ContinuationEndpoint) or not isinstance(
        candidate, ContinuationEndpoint
    ):
        raise TypeError("reference and candidate must be ContinuationEndpoint values")
    forced = ContinuationEndpoint.FORCED_VISIBLE_CONTROL
    if not allow_forced_control and (reference is forced or candidate is forced):
        raise ValueError("forced visible boundaries are not scientific endpoints")
    same_atom = reference is candidate
    visible_defined = reference in {
        ContinuationEndpoint.NATURAL_VISIBLE,
        forced,
    } and candidate in {
        ContinuationEndpoint.NATURAL_VISIBLE,
        forced,
    }
    return EndpointPairAudit(
        reference=reference,
        candidate=candidate,
        same_atom=same_atom,
        endpoint_kl=0.0 if same_atom else float("inf"),
        visible_continuation_defined=visible_defined,
        scientific_visible_pair=(
            reference is ContinuationEndpoint.NATURAL_VISIBLE
            and candidate is ContinuationEndpoint.NATURAL_VISIBLE
        ),
    )


def paired_path_kl(reference_logits: Tensor, candidate_logits: Tensor) -> PairedPathKL:
    """Compute categorical KL per shared-history step without smoothing.

    The caller is responsible for ensuring that both tensors were evaluated on
    exactly the same histories sampled from the reference policy.
    """

    if not isinstance(reference_logits, Tensor) or not isinstance(candidate_logits, Tensor):
        raise TypeError("reference_logits and candidate_logits must be tensors")
    if reference_logits.shape != candidate_logits.shape or reference_logits.ndim != 2:
        raise ValueError("logit tensors must have matching [steps, vocabulary] shapes")
    if reference_logits.shape[0] < 1 or reference_logits.shape[1] < 2:
        raise ValueError("at least one step and two vocabulary rows are required")
    if not reference_logits.is_floating_point() or not candidate_logits.is_floating_point():
        raise TypeError("logit tensors must be floating point")
    if not torch.isfinite(reference_logits).all() or not torch.isfinite(candidate_logits).all():
        raise ValueError("full-softmax KL requires finite logits")

    ref_log_probs = torch.log_softmax(reference_logits.to(torch.float64), dim=-1)
    cand_log_probs = torch.log_softmax(candidate_logits.to(torch.float64), dim=-1)
    per_step = torch.sum(ref_log_probs.exp() * (ref_log_probs - cand_log_probs), dim=-1)
    if torch.any(per_step < -1e-12):
        raise FloatingPointError("categorical KL is materially negative")
    # Roundoff at machine epsilon is represented as zero, not used to repair a
    # model-level negative or a support mismatch.
    per_step = torch.where(per_step.abs() <= 1e-12, torch.zeros_like(per_step), per_step)
    return PairedPathKL(per_step=per_step, total=float(per_step.sum().item()))
