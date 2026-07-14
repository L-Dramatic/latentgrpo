from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TopKConcreteSample:
    ordered_indices: Tensor
    weights: Tensor
    selected_scores: Tensor


def _positive_scalar(value: float | Tensor, reference: Tensor, name: str) -> Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.numel() != 1 or not torch.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


def _validate_logits(logits: Tensor) -> None:
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("logits must contain at least two categories")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")


def _broadcast_action(
    values: Tensor,
    ordered_indices: Tensor,
    logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_logits(logits)
    if not isinstance(values, Tensor) or not values.is_floating_point():
        raise TypeError("action values must be a floating-point tensor")
    if not isinstance(ordered_indices, Tensor) or ordered_indices.dtype != torch.long:
        raise TypeError("ordered_indices must be a torch.long tensor")
    if values.shape != ordered_indices.shape or values.ndim < 1:
        raise ValueError("action values and ordered indices must have matching shapes")
    top_k = values.shape[-1]
    vocabulary_size = logits.shape[-1]
    if not 1 <= top_k <= vocabulary_size:
        raise ValueError("top-k action width must fit within the vocabulary")
    batch_shape = torch.broadcast_shapes(values.shape[:-1], logits.shape[:-1])
    values = values.expand(batch_shape + (top_k,))
    ordered_indices = ordered_indices.expand(batch_shape + (top_k,))
    logits = logits.expand(batch_shape + (vocabulary_size,))
    if (ordered_indices < 0).any() or (ordered_indices >= vocabulary_size).any():
        raise ValueError("ordered indices are outside the vocabulary")
    sorted_indices = torch.sort(ordered_indices, dim=-1).values
    if top_k > 1 and (sorted_indices[..., 1:] == sorted_indices[..., :-1]).any():
        raise ValueError("top-k actions cannot repeat an index")
    if not torch.isfinite(values).all():
        raise ValueError("action values must be finite")
    return values, ordered_indices, logits


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
        if batch_shape != logits.shape[:-1]:
            logits = logits.expand(batch_shape + (logits.shape[-1],))
        mask = candidate_mask.expand(logits.shape)
    if (mask.sum(dim=-1) < top_k).any():
        raise ValueError("each candidate set must contain at least top_k entries")
    return mask


def _log1mexp(nonpositive: Tensor) -> Tensor:
    cutoff = -torch.log(torch.tensor(2.0, dtype=nonpositive.dtype, device=nonpositive.device))
    return torch.where(
        nonpositive < cutoff,
        torch.log1p(-torch.exp(nonpositive)),
        torch.log(-torch.expm1(nonpositive)),
    )


def _selected_and_unselected_log_mass(
    scaled_logits: Tensor,
    ordered_indices: Tensor,
    candidate_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    selected = torch.gather(scaled_logits, -1, ordered_indices)
    selected_mask = torch.gather(candidate_mask, -1, ordered_indices)
    masked_logits = scaled_logits.masked_fill(~candidate_mask, -torch.inf)
    selected_for_mass = selected.masked_fill(~selected_mask, -torch.inf)
    log_total = torch.logsumexp(masked_logits, dim=-1)
    log_selected = torch.logsumexp(selected_for_mass, dim=-1)
    log_selected_fraction = (log_selected - log_total).clamp(max=0.0)
    log_unselected = log_total + _log1mexp(log_selected_fraction)
    return selected, log_unselected, selected_mask.all(dim=-1)


def sample_topk_concrete(
    logits: Tensor,
    *,
    top_k: int,
    temperature: float | Tensor,
    gumbel_scale: float | Tensor = 1.0,
    candidate_mask: Tensor | None = None,
    sample_shape: Sequence[int] | torch.Size = (),
    generator: torch.Generator | None = None,
) -> TopKConcreteSample:
    """Sample the ordered top-k scores and their normalized executed weights."""

    _validate_logits(logits)
    if not 1 <= int(top_k) <= logits.shape[-1]:
        raise ValueError("top_k must fit within the vocabulary")
    temperature_tensor = _positive_scalar(temperature, logits, "temperature")
    scale_tensor = _positive_scalar(gumbel_scale, logits, "gumbel_scale")
    base_candidate_mask = _broadcast_candidate_mask(
        candidate_mask, logits, top_k=int(top_k)
    )
    shape = torch.Size(sample_shape) + logits.shape
    uniforms = torch.rand(
        shape,
        dtype=logits.dtype,
        device=logits.device,
        generator=generator,
    )
    epsilon = torch.finfo(logits.dtype).eps
    uniforms = uniforms.clamp(min=epsilon, max=1.0 - epsilon)
    gumbels = -torch.log(-torch.log(uniforms))
    scores = logits.expand(shape) + scale_tensor * gumbels
    scores = scores.masked_fill(~base_candidate_mask.expand(shape), -torch.inf)
    selected_scores, ordered_indices = torch.topk(
        scores, k=int(top_k), dim=-1, sorted=True
    )
    weights = torch.softmax(selected_scores / temperature_tensor, dim=-1)
    return TopKConcreteSample(
        ordered_indices=ordered_indices,
        weights=weights,
        selected_scores=selected_scores,
    )


def topk_concrete_log_density(
    weights: Tensor,
    ordered_indices: Tensor,
    logits: Tensor,
    *,
    temperature: float | Tensor,
    gumbel_scale: float | Tensor = 1.0,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    """Exact density of normalized ordered Gumbel top-k scores.

    The density is with respect to counting measure over ordered, distinct
    vocabulary indices and Lebesgue measure on the first ``k-1`` simplex
    coordinates. The support requires weights to follow the descending score
    order carried by ``ordered_indices``.
    """

    weights, ordered_indices, logits = _broadcast_action(
        weights, ordered_indices, logits
    )
    if (weights <= 0).any():
        raise ValueError("top-k weights must be strictly positive")
    sums = weights.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError("top-k weights must lie on the simplex")
    tolerance = 16 * torch.finfo(weights.dtype).eps
    if weights.shape[-1] > 1 and (
        weights[..., 1:] - weights[..., :-1] > tolerance
    ).any():
        raise ValueError("top-k weights must be in descending score order")

    temperature_tensor = _positive_scalar(temperature, logits, "temperature")
    scale_tensor = _positive_scalar(gumbel_scale, logits, "gumbel_scale")
    eta = temperature_tensor / scale_tensor
    scaled_logits = logits / scale_tensor
    expanded_candidate_mask = _broadcast_candidate_mask(
        candidate_mask, logits, top_k=weights.shape[-1]
    )
    selected_logits, log_unselected_mass, selection_is_valid = (
        _selected_and_unselected_log_mass(
            scaled_logits, ordered_indices, expanded_candidate_mask
        )
    )
    log_weights = torch.log(weights)
    denominator_terms = selected_logits - eta * log_weights
    unselected_term = log_unselected_mass - eta * log_weights[..., -1]
    denominator_terms = torch.cat(
        [denominator_terms, unselected_term.unsqueeze(-1)], dim=-1
    )
    log_denominator = torch.logsumexp(denominator_terms, dim=-1)
    top_k = weights.shape[-1]
    log_constant = (
        torch.lgamma(
            torch.tensor(float(top_k), dtype=weights.dtype, device=weights.device)
        )
        + (top_k - 1) * torch.log(eta)
    )
    log_numerator = (
        selected_logits - (eta + 1.0) * log_weights
    ).sum(dim=-1)
    log_density = log_constant + log_numerator - top_k * log_denominator
    return torch.where(
        selection_is_valid,
        log_density,
        torch.full_like(log_density, -torch.inf),
    )


def ordered_plackett_luce_log_probability(
    ordered_indices: Tensor,
    logits: Tensor,
    *,
    gumbel_scale: float | Tensor = 1.0,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    """Probability mass of the ordered support induced by Gumbel top-k."""

    if not isinstance(ordered_indices, Tensor) or ordered_indices.dtype != torch.long:
        raise TypeError("ordered_indices must be a torch.long tensor")
    if ordered_indices.ndim < 1:
        raise ValueError("ordered_indices must have an action dimension")
    dummy_values = torch.ones(
        ordered_indices.shape,
        dtype=logits.dtype,
        device=logits.device,
    )
    _, ordered_indices, logits = _broadcast_action(
        dummy_values, ordered_indices, logits
    )
    top_k = ordered_indices.shape[-1]
    scale_tensor = _positive_scalar(gumbel_scale, logits, "gumbel_scale")
    scaled_logits = logits / scale_tensor
    expanded_candidate_mask = _broadcast_candidate_mask(
        candidate_mask, logits, top_k=top_k
    )
    selected_mask = torch.gather(
        expanded_candidate_mask, -1, ordered_indices
    )
    selection_is_valid = selected_mask.all(dim=-1)
    masked_logits = scaled_logits.masked_fill(~expanded_candidate_mask, -torch.inf)
    selected_logits = torch.gather(scaled_logits, -1, ordered_indices)
    log_total = torch.logsumexp(masked_logits, dim=-1, keepdim=True)

    cumulative_selected = torch.logcumsumexp(selected_logits, dim=-1)
    no_removed_mass = torch.full_like(cumulative_selected[..., :1], -torch.inf)
    log_removed_before = torch.cat(
        [no_removed_mass, cumulative_selected[..., :-1]], dim=-1
    )
    log_removed_fraction = (log_removed_before - log_total).clamp(max=0.0)
    log_remaining_mass = log_total + _log1mexp(log_removed_fraction)
    log_probability = (selected_logits - log_remaining_mass).sum(dim=-1)
    return torch.where(
        selection_is_valid,
        log_probability,
        torch.full_like(log_probability, -torch.inf),
    )


def conditional_topk_weight_log_density(
    weights: Tensor,
    ordered_indices: Tensor,
    logits: Tensor,
    *,
    temperature: float | Tensor,
    gumbel_scale: float | Tensor = 1.0,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    """Exact density of top-k weights conditional on their ordered support."""

    joint = topk_concrete_log_density(
        weights,
        ordered_indices,
        logits,
        temperature=temperature,
        gumbel_scale=gumbel_scale,
        candidate_mask=candidate_mask,
    )
    support = ordered_plackett_luce_log_probability(
        ordered_indices,
        logits,
        gumbel_scale=gumbel_scale,
        candidate_mask=candidate_mask,
    )
    valid = torch.isfinite(joint) & torch.isfinite(support)
    conditional = joint - support
    return torch.where(
        valid,
        conditional,
        torch.full_like(conditional, -torch.inf),
    )


def naive_selected_gumbel_log_density(
    selected_scores: Tensor,
    ordered_indices: Tensor,
    logits: Tensor,
    *,
    gumbel_scale: float | Tensor = 1.0,
    reduction: str = "sum",
) -> Tensor:
    """Product-Gumbel surrogate that omits the top-k selection event."""

    selected_scores, ordered_indices, logits = _broadcast_action(
        selected_scores, ordered_indices, logits
    )
    scale_tensor = _positive_scalar(gumbel_scale, logits, "gumbel_scale")
    selected_logits = torch.gather(logits, -1, ordered_indices)
    margins = (selected_scores - selected_logits) / scale_tensor
    components = -torch.log(scale_tensor) - margins - torch.exp(-margins)
    if reduction == "sum":
        return components.sum(dim=-1)
    if reduction == "mean":
        return components.mean(dim=-1)
    raise ValueError("reduction must be 'sum' or 'mean'")


def selected_gumbel_log_density(
    selected_scores: Tensor,
    ordered_indices: Tensor,
    logits: Tensor,
    *,
    gumbel_scale: float | Tensor = 1.0,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    """Exact auxiliary density of labeled ordered top-k Gumbel scores."""

    selected_scores, ordered_indices, logits = _broadcast_action(
        selected_scores, ordered_indices, logits
    )
    tolerance = 16 * torch.finfo(selected_scores.dtype).eps
    if selected_scores.shape[-1] > 1 and (
        selected_scores[..., 1:] - selected_scores[..., :-1] > tolerance
    ).any():
        raise ValueError("selected scores must be sorted in descending order")
    scale_tensor = _positive_scalar(gumbel_scale, logits, "gumbel_scale")
    base = naive_selected_gumbel_log_density(
        selected_scores,
        ordered_indices,
        logits,
        gumbel_scale=scale_tensor,
        reduction="sum",
    )
    scaled_logits = logits / scale_tensor
    expanded_candidate_mask = _broadcast_candidate_mask(
        candidate_mask, logits, top_k=selected_scores.shape[-1]
    )
    _, log_unselected_mass, selection_is_valid = _selected_and_unselected_log_mass(
        scaled_logits, ordered_indices, expanded_candidate_mask
    )
    log_cdf_rate = log_unselected_mass - selected_scores[..., -1] / scale_tensor
    log_density = base - torch.exp(log_cdf_rate)
    return torch.where(
        selection_is_valid,
        log_density,
        torch.full_like(log_density, -torch.inf),
    )
