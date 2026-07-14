from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class ScoreSquashedTopKSample:
    ordered_indices: Tensor
    squashed_scores: Tensor
    weights: Tensor
    raw_selected_scores: Tensor


def _validate_logits(logits: Tensor) -> None:
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("logits must contain at least two categories")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")


def _positive_scalar(value: float | Tensor, reference: Tensor, name: str) -> Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.numel() != 1 or not torch.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


def _squash_parameters(
    lower: float | Tensor,
    upper: float | Tensor,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    lower_tensor = torch.as_tensor(
        lower, dtype=reference.dtype, device=reference.device
    )
    upper_tensor = torch.as_tensor(
        upper, dtype=reference.dtype, device=reference.device
    )
    if (
        lower_tensor.numel() != 1
        or upper_tensor.numel() != 1
        or not torch.isfinite(lower_tensor)
        or not torch.isfinite(upper_tensor)
        or lower_tensor >= upper_tensor
    ):
        raise ValueError("squash bounds must be finite scalars with lower < upper")
    midpoint = 0.5 * (lower_tensor + upper_tensor)
    half_range = 0.5 * (upper_tensor - lower_tensor)
    return midpoint, half_range


def score_squash(
    raw_scores: Tensor,
    *,
    lower: float | Tensor = -1.5,
    upper: float | Tensor = 3.0,
) -> Tensor:
    """Smooth monotone map from real scores to a fixed open interval."""

    if not isinstance(raw_scores, Tensor) or not raw_scores.is_floating_point():
        raise TypeError("raw_scores must be a floating-point tensor")
    midpoint, half_range = _squash_parameters(lower, upper, raw_scores)
    return midpoint + half_range * torch.tanh(raw_scores / half_range)


def inverse_score_squash(
    squashed_scores: Tensor,
    *,
    lower: float | Tensor = -1.5,
    upper: float | Tensor = 3.0,
) -> Tensor:
    """Inverse of :func:`score_squash` on the transform's open support."""

    if not isinstance(squashed_scores, Tensor) or not squashed_scores.is_floating_point():
        raise TypeError("squashed_scores must be a floating-point tensor")
    midpoint, half_range = _squash_parameters(lower, upper, squashed_scores)
    unit_scores = (squashed_scores - midpoint) / half_range
    if (unit_scores <= -1).any() or (unit_scores >= 1).any():
        raise ValueError("squashed scores must lie strictly inside the bounds")
    return half_range * torch.atanh(unit_scores)


def _broadcast_candidate_mask(
    candidate_mask: Tensor | None,
    logits: Tensor,
    *,
    top_k: int,
) -> Tensor:
    if candidate_mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
    else:
        if not isinstance(candidate_mask, Tensor) or candidate_mask.dtype != torch.bool:
            raise TypeError("candidate_mask must be a boolean tensor")
        if candidate_mask.ndim < 1 or candidate_mask.shape[-1] != logits.shape[-1]:
            raise ValueError("candidate_mask must match the vocabulary dimension")
        batch_shape = torch.broadcast_shapes(
            candidate_mask.shape[:-1], logits.shape[:-1]
        )
        mask = candidate_mask.expand(batch_shape + (logits.shape[-1],))
        logits = logits.expand(mask.shape)
    if (mask.sum(dim=-1) < top_k).any():
        raise ValueError("each candidate set must contain at least top_k entries")
    return mask


def _canonical_log_probabilities(logits: Tensor, candidate_mask: Tensor) -> Tensor:
    masked_logits = logits.masked_fill(~candidate_mask, -torch.inf)
    return masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)


def _log1mexp(nonpositive: Tensor) -> Tensor:
    cutoff = -torch.log(
        torch.tensor(2.0, dtype=nonpositive.dtype, device=nonpositive.device)
    )
    return torch.where(
        nonpositive < cutoff,
        torch.log1p(-torch.exp(nonpositive)),
        torch.log(-torch.expm1(nonpositive)),
    )


def _broadcast_action(
    scores: Tensor,
    ordered_indices: Tensor,
    logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_logits(logits)
    if not isinstance(scores, Tensor) or not scores.is_floating_point():
        raise TypeError("scores must be a floating-point tensor")
    if not isinstance(ordered_indices, Tensor) or ordered_indices.dtype != torch.long:
        raise TypeError("ordered_indices must be a torch.long tensor")
    if scores.shape != ordered_indices.shape or scores.ndim < 1:
        raise ValueError("scores and ordered_indices must have matching shapes")
    top_k = scores.shape[-1]
    vocabulary_size = logits.shape[-1]
    if not 1 <= top_k <= vocabulary_size:
        raise ValueError("top-k action width must fit within the vocabulary")
    batch_shape = torch.broadcast_shapes(scores.shape[:-1], logits.shape[:-1])
    scores = scores.expand(batch_shape + (top_k,))
    ordered_indices = ordered_indices.expand(batch_shape + (top_k,))
    logits = logits.expand(batch_shape + (vocabulary_size,))
    if (ordered_indices < 0).any() or (ordered_indices >= vocabulary_size).any():
        raise ValueError("ordered indices are outside the vocabulary")
    sorted_indices = torch.sort(ordered_indices, dim=-1).values
    if top_k > 1 and (sorted_indices[..., 1:] == sorted_indices[..., :-1]).any():
        raise ValueError("top-k actions cannot repeat an index")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return scores, ordered_indices, logits


def _selected_and_unselected_log_mass(
    scaled_log_probabilities: Tensor,
    ordered_indices: Tensor,
    candidate_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    selected = torch.gather(scaled_log_probabilities, -1, ordered_indices)
    selected_is_eligible = torch.gather(candidate_mask, -1, ordered_indices)
    masked = scaled_log_probabilities.masked_fill(~candidate_mask, -torch.inf)
    selected_for_mass = selected.masked_fill(~selected_is_eligible, -torch.inf)
    log_total = torch.logsumexp(masked, dim=-1)
    log_selected = torch.logsumexp(selected_for_mass, dim=-1)
    selected_fraction = (log_selected - log_total).clamp(max=0.0)
    log_unselected = log_total + _log1mexp(selected_fraction)
    return selected, log_unselected, selected_is_eligible.all(dim=-1)


def score_squashed_topk_from_gumbels(
    logits: Tensor,
    raw_gumbels: Tensor,
    *,
    top_k: int,
    temperature: float | Tensor,
    gumbel_scale: float | Tensor = 1.0,
    lower: float | Tensor = -1.5,
    upper: float | Tensor = 3.0,
    candidate_mask: Tensor | None = None,
) -> ScoreSquashedTopKSample:
    """Construct a score-squashed action from supplied standard Gumbels."""

    _validate_logits(logits)
    if not isinstance(raw_gumbels, Tensor) or not raw_gumbels.is_floating_point():
        raise TypeError("raw_gumbels must be a floating-point tensor")
    if raw_gumbels.shape[-1] != logits.shape[-1]:
        raise ValueError("raw_gumbels must match the vocabulary dimension")
    if raw_gumbels.dtype != logits.dtype or raw_gumbels.device != logits.device:
        raise ValueError("raw_gumbels and logits must share dtype and device")
    if not torch.isfinite(raw_gumbels).all():
        raise ValueError("raw_gumbels must be finite")
    if not 1 <= int(top_k) <= logits.shape[-1]:
        raise ValueError("top_k must fit within the vocabulary")
    temperature_tensor = _positive_scalar(temperature, logits, "temperature")
    scale_tensor = _positive_scalar(gumbel_scale, logits, "gumbel_scale")
    batch_shape = torch.broadcast_shapes(
        raw_gumbels.shape[:-1], logits.shape[:-1]
    )
    expanded_logits = logits.expand(batch_shape + (logits.shape[-1],))
    expanded_gumbels = raw_gumbels.expand(expanded_logits.shape)
    base_mask = _broadcast_candidate_mask(
        candidate_mask, logits, top_k=int(top_k)
    )
    expanded_mask = base_mask.expand(expanded_logits.shape)
    log_probabilities = _canonical_log_probabilities(
        expanded_logits, expanded_mask
    )
    raw_scores = log_probabilities + scale_tensor * expanded_gumbels
    raw_scores = raw_scores.masked_fill(~expanded_mask, -torch.inf)
    raw_selected_scores, ordered_indices = torch.topk(
        raw_scores, k=int(top_k), dim=-1, sorted=True
    )
    squashed_scores = score_squash(
        raw_selected_scores, lower=lower, upper=upper
    )
    weights = torch.softmax(squashed_scores / temperature_tensor, dim=-1)
    return ScoreSquashedTopKSample(
        ordered_indices=ordered_indices,
        squashed_scores=squashed_scores,
        weights=weights,
        raw_selected_scores=raw_selected_scores,
    )


def sample_score_squashed_topk(
    logits: Tensor,
    *,
    top_k: int,
    temperature: float | Tensor,
    gumbel_scale: float | Tensor = 1.0,
    lower: float | Tensor = -1.5,
    upper: float | Tensor = 3.0,
    candidate_mask: Tensor | None = None,
    sample_shape: Sequence[int] | torch.Size = (),
    generator: torch.Generator | None = None,
) -> ScoreSquashedTopKSample:
    """Sample an ordered support and smoothly bounded selected scores."""

    _validate_logits(logits)
    shape = torch.Size(sample_shape) + logits.shape
    uniforms = torch.rand(
        shape,
        dtype=logits.dtype,
        device=logits.device,
        generator=generator,
    )
    epsilon = torch.finfo(logits.dtype).eps
    uniforms = uniforms.clamp(min=epsilon, max=1.0 - epsilon)
    raw_gumbels = -torch.log(-torch.log(uniforms))
    return score_squashed_topk_from_gumbels(
        logits,
        raw_gumbels,
        top_k=top_k,
        temperature=temperature,
        gumbel_scale=gumbel_scale,
        lower=lower,
        upper=upper,
        candidate_mask=candidate_mask,
    )


def score_squashed_selected_log_density(
    squashed_scores: Tensor,
    ordered_indices: Tensor,
    logits: Tensor,
    *,
    gumbel_scale: float | Tensor = 1.0,
    lower: float | Tensor = -1.5,
    upper: float | Tensor = 3.0,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    """Exact density of ordered, score-squashed Gumbel top-k actions.

    Density is measured with counting measure over ordered supports and
    Lebesgue measure over every stored selected score. The action is augmented:
    downstream computation uses only its softmax mixture.
    """

    squashed_scores, ordered_indices, logits = _broadcast_action(
        squashed_scores, ordered_indices, logits
    )
    tolerance = 16 * torch.finfo(squashed_scores.dtype).eps
    if squashed_scores.shape[-1] > 1 and (
        squashed_scores[..., 1:] - squashed_scores[..., :-1] > tolerance
    ).any():
        raise ValueError("squashed scores must be sorted in descending order")
    scale_tensor = _positive_scalar(gumbel_scale, logits, "gumbel_scale")
    midpoint, half_range = _squash_parameters(lower, upper, squashed_scores)
    unit_scores = (squashed_scores - midpoint) / half_range
    if (unit_scores <= -1).any() or (unit_scores >= 1).any():
        raise ValueError("squashed scores must lie strictly inside the bounds")
    raw_selected_scores = half_range * torch.atanh(unit_scores)
    log_inverse_jacobian = -torch.log1p(-unit_scores.square())

    expanded_mask = _broadcast_candidate_mask(
        candidate_mask, logits, top_k=squashed_scores.shape[-1]
    )
    log_probabilities = _canonical_log_probabilities(logits, expanded_mask)
    scaled_log_probabilities = log_probabilities / scale_tensor
    selected_scaled_logits, log_unselected_mass, selection_is_valid = (
        _selected_and_unselected_log_mass(
            scaled_log_probabilities, ordered_indices, expanded_mask
        )
    )
    standardized_noise = (
        raw_selected_scores / scale_tensor - selected_scaled_logits
    )
    selected_log_density = (
        -torch.log(scale_tensor)
        - standardized_noise
        - torch.exp(-standardized_noise)
        + log_inverse_jacobian
    ).sum(dim=-1)
    log_cdf_rate = (
        log_unselected_mass - raw_selected_scores[..., -1] / scale_tensor
    )
    selection_log_probability = -torch.exp(log_cdf_rate)
    log_density = selected_log_density + selection_log_probability
    return torch.where(
        selection_is_valid,
        log_density,
        torch.full_like(log_density, -torch.inf),
    )
