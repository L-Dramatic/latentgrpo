from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


def average_ranks(values: Sequence[float] | Tensor) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    if tensor.numel() < 1 or not torch.isfinite(tensor).all():
        raise ValueError("rank values must be non-empty and finite")
    order = torch.argsort(tensor, stable=True)
    ranks = torch.empty_like(tensor)
    position = 0
    while position < tensor.numel():
        end = position + 1
        value = tensor[order[position]]
        while end < tensor.numel() and tensor[order[end]] == value:
            end += 1
        average = 0.5 * (position + end - 1)
        ranks[order[position:end]] = average
        position = end
    return ranks


def spearman_correlation(
    left: Sequence[float] | Tensor,
    right: Sequence[float] | Tensor,
) -> float:
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    if left_ranks.shape != right_ranks.shape:
        raise ValueError("ranked values must have identical shapes")
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(
        right_centered
    )
    if denominator <= 0:
        return 0.0
    return float((left_centered @ right_centered) / denominator)


def _top_indices(values: Tensor, fraction: float) -> Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    count = max(1, int(math.ceil(values.numel() * fraction)))
    return torch.argsort(values, descending=True, stable=True)[:count]


@dataclass(frozen=True)
class RankDiagnostic:
    spearman: float
    top_risk_recall: float
    hidden_top_risk_fraction: float
    count: int
    top_risk_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def rank_diagnostic(
    predictor: Sequence[float] | Tensor,
    target: Sequence[float] | Tensor,
    *,
    top_fraction: float = 0.2,
    predictor_screen_fraction: float = 0.5,
) -> RankDiagnostic:
    predictor_tensor = torch.as_tensor(predictor, dtype=torch.float64).flatten()
    target_tensor = torch.as_tensor(target, dtype=torch.float64).flatten()
    if predictor_tensor.shape != target_tensor.shape or predictor_tensor.numel() < 2:
        raise ValueError("predictor and target need matching lengths of at least two")
    if not torch.isfinite(predictor_tensor).all() or not torch.isfinite(
        target_tensor
    ).all():
        raise ValueError("diagnostic values must be finite")

    target_top = _top_indices(target_tensor, top_fraction)
    predictor_top = _top_indices(predictor_tensor, top_fraction)
    predictor_screen = _top_indices(predictor_tensor, predictor_screen_fraction)
    target_mask = torch.zeros(target_tensor.numel(), dtype=torch.bool)
    predictor_top_mask = torch.zeros_like(target_mask)
    predictor_screen_mask = torch.zeros_like(target_mask)
    target_mask[target_top] = True
    predictor_top_mask[predictor_top] = True
    predictor_screen_mask[predictor_screen] = True
    top_count = int(target_mask.sum())
    recall = float((target_mask & predictor_top_mask).sum()) / top_count
    hidden_fraction = float((target_mask & ~predictor_screen_mask).sum()) / top_count
    return RankDiagnostic(
        spearman=spearman_correlation(predictor_tensor, target_tensor),
        top_risk_recall=recall,
        hidden_top_risk_fraction=hidden_fraction,
        count=int(target_tensor.numel()),
        top_risk_count=top_count,
    )


def cumulative_prefix_kl(step_kl: Sequence[float] | Tensor, horizon: int) -> float:
    values = torch.as_tensor(step_kl, dtype=torch.float64).flatten()
    if horizon < 1 or horizon > values.numel():
        raise ValueError("prefix horizon must be within the recorded continuation")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("per-step KL values must be finite and non-negative")
    return float(values[:horizon].sum())


def continuation_rank_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix_horizons: Sequence[int],
    top_fraction: float = 0.2,
    predictor_screen_fraction: float = 0.5,
    include_coordinate_baselines: bool = True,
) -> dict[str, dict[str, float | int]]:
    """Compare cheap prefix predictors with full continuation KL.

    ``H=1`` keeps the conventional ``next_token_kl`` name. Longer prefixes are
    reported explicitly so a fixed formatting token cannot masquerade as a
    delayed behavioral effect.
    """

    if len(rows) < 2:
        raise ValueError("at least two candidate rows are required")
    horizons = [int(value) for value in prefix_horizons]
    if not horizons or sorted(set(horizons)) != horizons or horizons[0] != 1:
        raise ValueError("prefix_horizons must be sorted, unique, and begin at one")

    target = [float(row["total_kl"]) for row in rows]
    predictors: dict[str, list[float]] = {}
    for horizon in horizons:
        name = "next_token_kl" if horizon == 1 else f"prefix_kl_h{horizon}"
        predictors[name] = [
            cumulative_prefix_kl(row["mean_step_kl"], horizon) for row in rows
        ]

    if include_coordinate_baselines:
        for name in ("euclidean", "cosine", "diagonal_mahalanobis"):
            predictors[name] = [
                float(row["coordinate_distances"][name]) for row in rows
            ]

    return {
        name: rank_diagnostic(
            values,
            target,
            top_fraction=top_fraction,
            predictor_screen_fraction=predictor_screen_fraction,
        ).to_dict()
        for name, values in predictors.items()
    }
