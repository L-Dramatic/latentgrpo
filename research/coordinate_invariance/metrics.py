from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import Tensor


DivergenceName = Literal["kl", "reverse_kl", "symmetric_kl", "js"]


def _validate_logits(reference_logits: Tensor, candidate_logits: Tensor) -> None:
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError("reference and candidate logits must have identical shapes")
    if reference_logits.ndim < 1 or reference_logits.shape[-1] < 2:
        raise ValueError("logits must have at least two classes in the last dimension")
    if not reference_logits.is_floating_point() or not candidate_logits.is_floating_point():
        raise TypeError("logits must use floating-point dtypes")
    if not torch.isfinite(reference_logits).all() or not torch.isfinite(
        candidate_logits
    ).all():
        raise ValueError("logits must contain only finite values")


def categorical_kl_from_logits(reference_logits: Tensor, candidate_logits: Tensor) -> Tensor:
    """Compute ``KL(reference || candidate)`` without materializing raw probabilities twice."""

    _validate_logits(reference_logits, candidate_logits)
    reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
    candidate_log_probs = torch.log_softmax(candidate_logits, dim=-1)
    reference_probs = reference_log_probs.exp()
    return (reference_probs * (reference_log_probs - candidate_log_probs)).sum(dim=-1)


def stable_categorical_kl_from_logits(
    reference_logits: Tensor, candidate_logits: Tensor
) -> Tensor:
    """Compute KL accurately when two categorical distributions are very close.

    The usual log-softmax expression subtracts first-order terms and can round a
    small positive KL to zero. For nearby logits, this implementation centers
    their difference under the reference distribution and evaluates the
    non-negative ``expm1(x) - x`` remainder. Larger differences use the usual
    log-softmax expression to avoid exponent overflow.
    """

    _validate_logits(reference_logits, candidate_logits)
    reference64 = reference_logits.to(torch.float64)
    candidate64 = candidate_logits.to(torch.float64)
    reference_log_probs = torch.log_softmax(reference64, dim=-1)
    reference_probs = reference_log_probs.exp()
    delta = candidate64 - reference64
    mean_delta = (reference_probs * delta).sum(dim=-1, keepdim=True)
    centered = delta - mean_delta

    clipped = centered.clamp(min=-0.1, max=0.1)
    square = clipped.square()
    series_remainder = square * (
        0.5
        + clipped
        * (
            1.0 / 6.0
            + clipped
            * (1.0 / 24.0 + clipped * (1.0 / 120.0 + clipped / 720.0))
        )
    )
    direct_remainder = torch.expm1(clipped) - clipped
    remainder = torch.where(
        clipped.abs() <= 0.01, series_remainder, direct_remainder
    )
    excess_moment = (reference_probs * remainder).sum(dim=-1).clamp_min(0.0)
    nearby_kl = torch.log1p(excess_moment)

    candidate_log_probs = torch.log_softmax(candidate64, dim=-1)
    ordinary_kl = (
        reference_probs * (reference_log_probs - candidate_log_probs)
    ).sum(dim=-1).clamp_min(0.0)
    nearby = centered.abs().amax(dim=-1) <= 0.1
    return torch.where(nearby, nearby_kl, ordinary_kl)


def categorical_divergence_from_logits(
    reference_logits: Tensor,
    candidate_logits: Tensor,
    *,
    divergence: DivergenceName = "symmetric_kl",
) -> Tensor:
    _validate_logits(reference_logits, candidate_logits)
    if divergence == "kl":
        return categorical_kl_from_logits(reference_logits, candidate_logits)
    if divergence == "reverse_kl":
        return categorical_kl_from_logits(candidate_logits, reference_logits)
    if divergence == "symmetric_kl":
        forward = categorical_kl_from_logits(reference_logits, candidate_logits)
        reverse = categorical_kl_from_logits(candidate_logits, reference_logits)
        return 0.5 * (forward + reverse)
    if divergence == "js":
        reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
        candidate_log_probs = torch.log_softmax(candidate_logits, dim=-1)
        mixture_log_probs = torch.logsumexp(
            torch.stack([reference_log_probs, candidate_log_probs], dim=0), dim=0
        ) - torch.log(
            torch.tensor(
                2.0,
                dtype=reference_logits.dtype,
                device=reference_logits.device,
            )
        )
        reference_probs = reference_log_probs.exp()
        candidate_probs = candidate_log_probs.exp()
        reference_kl = (
            reference_probs * (reference_log_probs - mixture_log_probs)
        ).sum(dim=-1)
        candidate_kl = (
            candidate_probs * (candidate_log_probs - mixture_log_probs)
        ).sum(dim=-1)
        return 0.5 * (reference_kl + candidate_kl)
    raise ValueError(f"unsupported divergence: {divergence}")


def multi_horizon_divergence(
    reference_logits: Tensor,
    candidate_logits: Tensor,
    *,
    horizon_weights: Sequence[float] | Tensor | None = None,
    divergence: DivergenceName = "symmetric_kl",
) -> Tensor:
    """Aggregate categorical divergence over the penultimate horizon dimension.

    Inputs have shape ``(..., horizon, classes)``. The returned tensor has shape
    ``(...)``. Horizon weights are normalized to sum to one.
    """

    _validate_logits(reference_logits, candidate_logits)
    if reference_logits.ndim < 2:
        raise ValueError("multi-horizon logits must have shape (..., horizon, classes)")
    horizon = int(reference_logits.shape[-2])
    if horizon < 1:
        raise ValueError("horizon must be positive")

    if horizon_weights is None:
        weights = torch.ones(
            horizon,
            dtype=reference_logits.dtype,
            device=reference_logits.device,
        )
    else:
        weights = torch.as_tensor(
            horizon_weights,
            dtype=reference_logits.dtype,
            device=reference_logits.device,
        )
        if weights.shape != (horizon,):
            raise ValueError(f"horizon_weights must have shape ({horizon},)")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("horizon_weights must be finite and non-negative")
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("at least one horizon weight must be positive")
    normalized_weights = weights / weight_sum

    per_step = categorical_divergence_from_logits(
        reference_logits, candidate_logits, divergence=divergence
    )
    return (per_step * normalized_weights).sum(dim=-1)
