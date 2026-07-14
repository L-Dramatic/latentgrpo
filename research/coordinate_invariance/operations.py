from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .charts import LatentChart


DistanceName = Literal["euclidean", "cosine"]


@dataclass(frozen=True)
class NeighborResult:
    index: int
    distance: float
    all_distances: Tensor


def _validate_pair(left: Tensor, right: Tensor) -> None:
    if left.shape[-1] != right.shape[-1]:
        raise ValueError("latent tensors must have the same last dimension")
    if not left.is_floating_point() or not right.is_floating_point():
        raise TypeError("latent tensors must use floating-point dtypes")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("latent tensors must contain only finite values")


def coordinate_distances(
    query_native: Tensor,
    candidates_native: Tensor,
    chart: LatentChart,
    *,
    metric: DistanceName = "euclidean",
    epsilon: float = 1e-12,
) -> Tensor:
    """Measure query-to-candidate distance in a selected coordinate chart."""

    if query_native.shape != (chart.dimension,):
        raise ValueError(f"query_native must have shape ({chart.dimension},)")
    if candidates_native.ndim != 2 or candidates_native.shape[-1] != chart.dimension:
        raise ValueError(
            f"candidates_native must have shape (n, {chart.dimension})"
        )
    if candidates_native.shape[0] == 0:
        raise ValueError("at least one candidate is required")
    _validate_pair(query_native, candidates_native)

    query = chart.encode(query_native)
    candidates = chart.encode(candidates_native)
    if metric == "euclidean":
        return torch.linalg.vector_norm(candidates - query.unsqueeze(0), dim=-1)
    if metric == "cosine":
        query_norm = torch.linalg.vector_norm(query).clamp_min(epsilon)
        candidate_norm = torch.linalg.vector_norm(candidates, dim=-1).clamp_min(epsilon)
        similarities = (candidates @ query) / (candidate_norm * query_norm)
        return 1.0 - similarities.clamp(-1.0, 1.0)
    raise ValueError(f"unsupported metric: {metric}")


def nearest_neighbor_in_chart(
    query_native: Tensor,
    candidates_native: Tensor,
    chart: LatentChart,
    *,
    metric: DistanceName = "euclidean",
) -> NeighborResult:
    distances = coordinate_distances(
        query_native, candidates_native, chart, metric=metric
    )
    index = int(torch.argmin(distances).item())
    return NeighborResult(
        index=index,
        distance=float(distances[index].detach().cpu()),
        all_distances=distances,
    )


def add_isotropic_noise_in_chart(
    native: Tensor,
    chart: LatentChart,
    *,
    standard_deviation: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Add isotropic Gaussian noise in chart coordinates and return native values."""

    if standard_deviation < 0 or not torch.isfinite(
        torch.tensor(standard_deviation)
    ):
        raise ValueError("standard_deviation must be finite and non-negative")
    charted = chart.encode(native)
    noise = torch.randn(
        charted.shape,
        dtype=charted.dtype,
        device=charted.device,
        generator=generator,
    )
    return chart.decode(charted + standard_deviation * noise)


def interpolate_in_chart(
    left_native: Tensor,
    right_native: Tensor,
    chart: LatentChart,
    *,
    alpha: float,
) -> Tensor:
    """Interpolate coordinate values in a chart and decode to native space."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if left_native.shape != right_native.shape:
        raise ValueError("left_native and right_native must have identical shapes")
    _validate_pair(left_native, right_native)
    left_charted = chart.encode(left_native)
    right_charted = chart.encode(right_native)
    mixed = (1.0 - alpha) * left_charted + alpha * right_charted
    return chart.decode(mixed)

