from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from .metrics import stable_categorical_kl_from_logits


@dataclass(frozen=True)
class PrefixGeometry:
    objective_gradient: Tensor | None
    cumulative_metrics: tuple[Tensor, ...]


def consequential_basis(
    full_gradient: Tensor,
    *,
    dimension: int,
    seed: int,
) -> Tensor:
    if full_gradient.ndim != 1 or not full_gradient.is_floating_point():
        raise ValueError("full_gradient must be a floating-point vector")
    if not torch.isfinite(full_gradient).all():
        raise ValueError("full_gradient must be finite")
    ambient = int(full_gradient.numel())
    if not 1 <= dimension <= ambient:
        raise ValueError("dimension must lie within the ambient dimension")
    gradient64 = full_gradient.detach().to(device="cpu", dtype=torch.float64)
    norm = torch.linalg.vector_norm(gradient64)
    if float(norm) <= 0.0:
        raise ValueError("full_gradient must be nonzero")
    first = gradient64 / norm
    if dimension == 1:
        return first.unsqueeze(1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random = torch.randn(
        ambient,
        dimension - 1,
        generator=generator,
        dtype=torch.float64,
    )
    random = random - first.unsqueeze(1) @ (first.unsqueeze(0) @ random)
    complement, _ = torch.linalg.qr(random, mode="reduced")
    complement = complement - first.unsqueeze(1) @ (
        first.unsqueeze(0) @ complement
    )
    complement, _ = torch.linalg.qr(complement, mode="reduced")
    return torch.cat([first.unsqueeze(1), complement], dim=1)


def prefix_geometry(
    logits: Tensor,
    jacobian: Tensor,
    *,
    target_ids: Tensor | Sequence[int] | None = None,
    objective_horizon: int | None = None,
) -> PrefixGeometry:
    if logits.ndim != 2 or jacobian.ndim != 3:
        raise ValueError("logits and jacobian must have shapes (H,V) and (H,V,D)")
    if logits.shape[:2] != jacobian.shape[:2]:
        raise ValueError("logits and jacobian horizon/vocabulary shapes must match")
    horizon, _, dimension = jacobian.shape
    if horizon < 1 or dimension < 1:
        raise ValueError("horizon and subspace dimension must be positive")
    logits64 = logits.to(torch.float64)
    jacobian64 = jacobian.to(torch.float64)
    targets = None
    if target_ids is not None:
        targets = torch.as_tensor(target_ids, dtype=torch.long, device=logits.device)
        if targets.shape != (horizon,):
            raise ValueError("target_ids must match the full horizon")
        if objective_horizon is None:
            objective_horizon = horizon
        if not 1 <= int(objective_horizon) <= horizon:
            raise ValueError("objective_horizon lies outside the measured horizon")
    elif objective_horizon is not None:
        raise ValueError("objective_horizon requires target_ids")

    cumulative = torch.zeros(
        (dimension, dimension), dtype=torch.float64, device=logits.device
    )
    metrics: list[Tensor] = []
    gradient = torch.zeros(dimension, dtype=torch.float64, device=logits.device)
    for step in range(horizon):
        probabilities = torch.softmax(logits64[step], dim=0)
        current = jacobian64[step]
        mean = probabilities @ current
        weighted = probabilities.unsqueeze(-1) * current
        fisher = current.transpose(0, 1) @ weighted - torch.outer(mean, mean)
        cumulative = cumulative + 0.5 * (fisher + fisher.transpose(0, 1))
        metrics.append(cumulative.clone())
        if targets is not None and step < int(objective_horizon):
            gradient = gradient + current[targets[step]] - mean
    if targets is not None:
        gradient = gradient / int(objective_horizon)
    return PrefixGeometry(
        objective_gradient=gradient if targets is not None else None,
        cumulative_metrics=tuple(metrics),
    )


def fit_diagonal_whitening_precision(
    states: Tensor,
    *,
    variance_floor_fraction_of_median: float,
) -> tuple[Tensor, float]:
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("states must contain at least two activation vectors")
    if not states.is_floating_point() or not torch.isfinite(states).all():
        raise ValueError("states must be finite floating-point values")
    if (
        not math.isfinite(variance_floor_fraction_of_median)
        or variance_floor_fraction_of_median <= 0.0
    ):
        raise ValueError("variance floor fraction must be finite and positive")
    variance = states.to(torch.float64).var(dim=0, unbiased=False)
    positive = variance[variance > 0]
    if positive.numel() == 0:
        raise ValueError("calibration states have zero variance in every coordinate")
    floor = float(torch.median(positive) * variance_floor_fraction_of_median)
    precision = torch.reciprocal(variance + floor)
    if not torch.isfinite(precision).all() or (precision <= 0).any():
        raise ValueError("whitening precision is not positive and finite")
    return precision, floor


def projected_diagonal_metric(basis: Tensor, precision: Tensor) -> Tensor:
    if basis.ndim != 2 or precision.ndim != 1:
        raise ValueError("basis and precision must have shapes (D,d) and (D,)")
    if basis.shape[0] != precision.numel():
        raise ValueError("basis and precision ambient dimensions differ")
    basis64 = basis.to(torch.float64)
    precision64 = precision.to(device=basis.device, dtype=torch.float64)
    metric = basis64.transpose(0, 1) @ (precision64.unsqueeze(1) * basis64)
    return 0.5 * (metric + metric.transpose(0, 1))


def regularize_with_whitening(
    metric: Tensor,
    whitening_metric: Tensor,
    *,
    relative_generalized_ridge: float,
) -> tuple[Tensor, float]:
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("metric must be square")
    if whitening_metric.shape != metric.shape:
        raise ValueError("whitening metric shape differs from metric")
    if not math.isfinite(relative_generalized_ridge) or relative_generalized_ridge <= 0:
        raise ValueError("relative generalized ridge must be finite and positive")
    metric64 = 0.5 * (
        metric.to(torch.float64) + metric.to(torch.float64).transpose(0, 1)
    )
    whitening64 = 0.5 * (
        whitening_metric.to(device=metric64.device, dtype=torch.float64)
        + whitening_metric.to(device=metric64.device, dtype=torch.float64).transpose(0, 1)
    )
    dimension = int(metric64.shape[0])
    generalized_mean = float(
        torch.trace(torch.linalg.solve(whitening64, metric64)) / dimension
    )
    if not math.isfinite(generalized_mean) or generalized_mean <= 0.0:
        raise ValueError("metric has no positive generalized sensitivity")
    ridge = float(relative_generalized_ridge) * generalized_mean
    regularized = metric64 + ridge * whitening64
    if float(torch.linalg.eigvalsh(regularized).min()) <= 0.0:
        raise ValueError("regularized metric is not positive definite")
    return regularized, ridge


def normalize_to_predicted_gain(
    direction: Tensor,
    objective_gradient: Tensor,
    *,
    predicted_gain: float,
) -> Tensor:
    if direction.ndim != 1 or objective_gradient.shape != direction.shape:
        raise ValueError("direction and objective gradient must be matching vectors")
    if not math.isfinite(predicted_gain) or predicted_gain <= 0.0:
        raise ValueError("predicted_gain must be finite and positive")
    denominator = torch.dot(objective_gradient, direction)
    if not torch.isfinite(denominator) or float(denominator) <= 0.0:
        raise ValueError("direction must have positive predicted objective gain")
    return direction * (predicted_gain / denominator)


def metric_update(
    metric: Tensor,
    objective_gradient: Tensor,
    *,
    predicted_gain: float,
) -> Tensor:
    direction = torch.linalg.solve(metric, objective_gradient)
    return normalize_to_predicted_gain(
        direction, objective_gradient, predicted_gain=predicted_gain
    )


def chart_euclidean_update(
    objective_gradient: Tensor,
    chart_matrix: Tensor,
    *,
    predicted_gain: float,
) -> Tensor:
    if chart_matrix.shape != (
        objective_gradient.numel(),
        objective_gradient.numel(),
    ):
        raise ValueError("chart matrix has the wrong shape")
    chart_gradient = torch.linalg.solve(
        chart_matrix.transpose(0, 1), objective_gradient
    )
    native_direction = torch.linalg.solve(chart_matrix, chart_gradient)
    return normalize_to_predicted_gain(
        native_direction,
        objective_gradient,
        predicted_gain=predicted_gain,
    )


def semantic_prefix_horizon(
    tokenizer: Any,
    token_ids: Sequence[int] | Tensor,
    *,
    minimum_tokens: int,
    maximum_tokens: int,
    boundary_regex: str,
) -> int:
    ids = [int(value) for value in torch.as_tensor(token_ids).flatten().tolist()]
    maximum = min(int(maximum_tokens), len(ids))
    if not 1 <= int(minimum_tokens) <= maximum:
        raise ValueError("semantic prefix bounds are invalid")
    pattern = re.compile(boundary_regex)
    for horizon in range(int(minimum_tokens), maximum + 1):
        text = tokenizer.decode(ids[:horizon], skip_special_tokens=False)
        if pattern.search(text):
            return horizon
    return maximum


def mean_factual_log_probability(
    logits: Tensor,
    target_ids: Tensor | Sequence[int],
    *,
    start: int,
    end: int,
) -> float:
    targets = torch.as_tensor(target_ids, dtype=torch.long, device=logits.device)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("logits and target_ids have incompatible shapes")
    if not 0 <= start < end <= logits.shape[0]:
        raise ValueError("utility interval is invalid")
    selected_logits = logits[start:end].to(torch.float64)
    selected_targets = targets[start:end]
    rows = torch.arange(end - start, device=logits.device)
    value = torch.log_softmax(selected_logits, dim=-1)[rows, selected_targets].mean()
    return float(value.detach().cpu())


def summed_categorical_kl(
    reference_logits: Tensor,
    candidate_logits: Tensor,
    *,
    start: int,
    end: int,
    chunk_size: int = 8,
) -> float:
    if reference_logits.shape != candidate_logits.shape or reference_logits.ndim != 2:
        raise ValueError("reference and candidate logits must have matching (H,V) shapes")
    if not 0 <= start < end <= reference_logits.shape[0]:
        raise ValueError("KL interval is invalid")
    total = torch.zeros((), dtype=torch.float64, device=reference_logits.device)
    for chunk_start in range(start, end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end)
        total = total + stable_categorical_kl_from_logits(
            reference_logits[chunk_start:chunk_end],
            candidate_logits[chunk_start:chunk_end],
        ).sum()
    return float(total.detach().cpu())


def _average_ranks(values: Sequence[float]) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.ndim != 1 or tensor.numel() < 2 or not torch.isfinite(tensor).all():
        raise ValueError("rank input must contain at least two finite values")
    order = torch.argsort(tensor, stable=True)
    ranks = torch.empty_like(tensor)
    position = 0
    while position < tensor.numel():
        end = position + 1
        value = tensor[order[position]]
        while end < tensor.numel() and tensor[order[end]] == value:
            end += 1
        average = 0.5 * ((position + 1) + end)
        ranks[order[position:end]] = average
        position = end
    return ranks


def spearman_correlation(predicted: Sequence[float], actual: Sequence[float]) -> float:
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual lengths differ")
    left = _average_ranks(predicted)
    right = _average_ranks(actual)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def top_risk_recall(
    predicted: Sequence[float],
    actual: Sequence[float],
    *,
    fraction: float,
) -> float:
    if len(predicted) != len(actual) or len(predicted) < 2:
        raise ValueError("predicted and actual must have equal nontrivial length")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0,1]")
    count = max(1, int(math.ceil(fraction * len(predicted))))
    predicted_top = set(
        torch.argsort(torch.as_tensor(predicted), descending=True)[:count].tolist()
    )
    actual_top = set(
        torch.argsort(torch.as_tensor(actual), descending=True)[:count].tolist()
    )
    return len(predicted_top & actual_top) / count
