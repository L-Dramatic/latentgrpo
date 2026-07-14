from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ScoreMeanDiagnostics:
    mean: Tensor
    mean_l2: float
    rms_standard_error: float
    signal_to_noise: float


def _require_floating_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def importance_weight_mean(old_log_density: Tensor, new_log_density: Tensor) -> float:
    """Estimate E_old[p_new(a) / p_old(a)]."""

    _require_floating_tensor(old_log_density, "old_log_density")
    _require_floating_tensor(new_log_density, "new_log_density")
    if old_log_density.shape != new_log_density.shape:
        raise ValueError("log-density tensors must have identical shapes")
    return float(torch.exp(new_log_density - old_log_density).mean())


def score_mean_diagnostics(scores: Tensor) -> ScoreMeanDiagnostics:
    """Summarize the zero-mean score identity on Monte Carlo samples."""

    _require_floating_tensor(scores, "scores")
    if scores.ndim != 2 or scores.shape[0] < 2:
        raise ValueError("scores must have shape (samples, parameters)")
    mean = scores.mean(dim=0)
    centered = scores - mean
    mean_square_error = centered.square().sum(dim=-1).mean() / scores.shape[0]
    standard_error = math.sqrt(float(mean_square_error))
    mean_l2 = float(mean.norm())
    signal_to_noise = mean_l2 / max(standard_error, torch.finfo(scores.dtype).tiny)
    return ScoreMeanDiagnostics(
        mean=mean,
        mean_l2=mean_l2,
        rms_standard_error=standard_error,
        signal_to_noise=signal_to_noise,
    )


def aggregate_log_ratios(log_ratios: Tensor, mask: Tensor, *, reduction: str) -> Tensor:
    """Aggregate component or token log ratios with an explicit convention."""

    _require_floating_tensor(log_ratios, "log_ratios")
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean tensor")
    if mask.shape != log_ratios.shape:
        raise ValueError("mask and log_ratios must have identical shapes")
    counts = mask.sum(dim=-1)
    if (counts == 0).any():
        raise ValueError("every row must include at least one ratio")
    totals = torch.where(mask, log_ratios, 0.0).sum(dim=-1)
    if reduction == "sum":
        return totals
    if reduction == "mean":
        return totals / counts
    raise ValueError("reduction must be 'sum' or 'mean'")


def ppo_clip_mask(log_ratio: Tensor, advantage: Tensor, *, epsilon: float) -> Tensor:
    """Return whether PPO clipping changes the active objective branch."""

    _require_floating_tensor(log_ratio, "log_ratio")
    _require_floating_tensor(advantage, "advantage")
    if log_ratio.shape != advantage.shape:
        advantage = torch.broadcast_to(advantage, log_ratio.shape)
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    upper = math.log1p(epsilon)
    lower = math.log1p(-epsilon)
    return torch.where(advantage >= 0, log_ratio > upper, log_ratio < lower)
