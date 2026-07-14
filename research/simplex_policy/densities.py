from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor


def _validate_matching_floats(
    left: Tensor, right: Tensor, *, minimum_last_dimension: int = 2
) -> None:
    if not isinstance(left, Tensor) or not isinstance(right, Tensor):
        raise TypeError("inputs must be tensors")
    if left.shape != right.shape:
        raise ValueError("inputs must have identical shapes")
    if left.ndim < 1 or left.shape[-1] < minimum_last_dimension:
        raise ValueError(
            "the final dimension is smaller than the required minimum"
        )
    if not left.is_floating_point() or not right.is_floating_point():
        raise TypeError("inputs must use floating-point dtypes")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("inputs must contain only finite values")


def _temperature_tensor(temperature: float | Tensor, reference: Tensor) -> Tensor:
    value = torch.as_tensor(
        temperature, dtype=reference.dtype, device=reference.device
    )
    if value.numel() != 1 or not torch.isfinite(value) or value <= 0:
        raise ValueError("temperature must be a finite positive scalar")
    return value.reshape(())


def _validate_simplex(sample: Tensor, *, tolerance: float = 1e-6) -> None:
    if not isinstance(sample, Tensor) or not sample.is_floating_point():
        raise TypeError("sample must be a floating-point tensor")
    if sample.ndim < 1 or sample.shape[-1] < 2:
        raise ValueError("sample must contain at least two categories")
    if not torch.isfinite(sample).all() or (sample <= 0).any():
        raise ValueError("Concrete samples must be finite and strictly positive")
    sums = sample.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=tolerance, rtol=tolerance):
        raise ValueError("sample must lie on the probability simplex")


def sample_concrete(
    logits: Tensor,
    *,
    temperature: float | Tensor,
    sample_shape: Sequence[int] | torch.Size = (),
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample a Concrete action and return it with its auxiliary perturbed scores."""

    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("logits must contain at least two categories")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    temperature_tensor = _temperature_tensor(temperature, logits)
    normalized_sample_shape = torch.Size(sample_shape)
    shape = normalized_sample_shape + logits.shape
    uniforms = torch.rand(
        shape,
        dtype=logits.dtype,
        device=logits.device,
        generator=generator,
    )
    epsilon = torch.finfo(logits.dtype).eps
    uniforms = uniforms.clamp(min=epsilon, max=1.0 - epsilon)
    gumbels = -torch.log(-torch.log(uniforms))
    perturbed_scores = logits + gumbels
    sample = torch.softmax(perturbed_scores / temperature_tensor, dim=-1)
    return sample, perturbed_scores


def concrete_log_density(
    sample: Tensor,
    logits: Tensor,
    *,
    temperature: float | Tensor,
) -> Tensor:
    """Evaluate the exact Concrete density on the open probability simplex."""

    _validate_matching_floats(sample, logits)
    _validate_simplex(sample)
    temperature_tensor = _temperature_tensor(temperature, sample)
    category_count = int(sample.shape[-1])
    log_sample = torch.log(sample)
    log_normalizer = (
        torch.lgamma(
            torch.tensor(
                float(category_count), dtype=sample.dtype, device=sample.device
            )
        )
        + (category_count - 1) * torch.log(temperature_tensor)
    )
    numerator = (logits - (temperature_tensor + 1.0) * log_sample).sum(dim=-1)
    denominator = category_count * torch.logsumexp(
        logits - temperature_tensor * log_sample, dim=-1
    )
    return log_normalizer + numerator - denominator


def auxiliary_gumbel_log_density(
    perturbed_scores: Tensor,
    logits: Tensor,
    *,
    scale: float | Tensor = 1.0,
) -> Tensor:
    """Evaluate the product Gumbel density of auxiliary perturbed scores."""

    _validate_matching_floats(perturbed_scores, logits)
    scale_tensor = torch.as_tensor(
        scale, dtype=logits.dtype, device=logits.device
    )
    if scale_tensor.numel() != 1 or not torch.isfinite(scale_tensor) or scale_tensor <= 0:
        raise ValueError("scale must be a finite positive scalar")
    standardized = (perturbed_scores - logits) / scale_tensor
    component_log_density = (
        -standardized - torch.exp(-standardized) - torch.log(scale_tensor)
    )
    return component_log_density.sum(dim=-1)


def concrete_score(
    sample: Tensor,
    logits: Tensor,
    *,
    temperature: float | Tensor,
) -> Tensor:
    """Analytic score of the Concrete log density with respect to logits."""

    _validate_matching_floats(sample, logits)
    _validate_simplex(sample)
    temperature_tensor = _temperature_tensor(temperature, sample)
    category_count = int(sample.shape[-1])
    posterior_weights = torch.softmax(
        logits - temperature_tensor * torch.log(sample), dim=-1
    )
    return torch.ones_like(posterior_weights) - category_count * posterior_weights


def auxiliary_gumbel_score(perturbed_scores: Tensor, logits: Tensor) -> Tensor:
    """Analytic score of independent unit-scale Gumbel locations."""

    _validate_matching_floats(perturbed_scores, logits)
    return 1.0 - torch.exp(-(perturbed_scores - logits))


def effective_sample_size(log_weights: Tensor) -> float:
    if not isinstance(log_weights, Tensor) or not log_weights.is_floating_point():
        raise TypeError("log_weights must be a floating-point tensor")
    if log_weights.ndim != 1 or log_weights.numel() < 1:
        raise ValueError("log_weights must be a non-empty vector")
    if not torch.isfinite(log_weights).all():
        raise ValueError("log_weights must be finite")
    normalized = torch.softmax(log_weights, dim=0)
    return float(1.0 / normalized.square().sum())


def ppo_clip_mask(
    log_ratio: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float,
) -> Tensor:
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must lie in (0, 1)")
    _validate_matching_floats(log_ratio, advantages)
    ratio = torch.exp(log_ratio)
    return ((advantages >= 0) & (ratio > 1.0 + clip_epsilon)) | (
        (advantages < 0) & (ratio < 1.0 - clip_epsilon)
    )


def latent_grpo_surrogate_log_prob(
    current_log_probs: Tensor,
    rollout_perturbed_scores: Tensor,
    advantages: Tensor | None = None,
) -> Tensor:
    """Reproduce the inspected Latent-GRPO component surrogate and its STE path."""

    _validate_matching_floats(
        current_log_probs,
        rollout_perturbed_scores,
        minimum_last_dimension=1,
    )
    raw_margin = rollout_perturbed_scores - current_log_probs
    standard = -raw_margin - torch.exp(-raw_margin)
    if advantages is None:
        return standard
    if not isinstance(advantages, Tensor) or not advantages.is_floating_point():
        raise TypeError("advantages must be a floating-point tensor")
    try:
        expanded_advantages = torch.broadcast_to(advantages, raw_margin.shape)
    except RuntimeError as error:
        raise ValueError("advantages cannot be broadcast to the margin shape") from error
    if not torch.isfinite(expanded_advantages).all():
        raise ValueError("advantages must be finite")
    flip_mask = (expanded_advantages <= 0) & (raw_margin < 0)
    flipped_margin = -raw_margin
    flipped = -flipped_margin - torch.exp(-flipped_margin)
    proxy = torch.where(flip_mask, flipped, standard)
    return standard.detach() + (proxy - proxy.detach())
