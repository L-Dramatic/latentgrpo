from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FunctionalTrustRegionStep:
    step: Tensor
    direction: Tensor
    scale: float
    predicted_gain: float
    trust_cost: float
    metric_condition_number: float


def _validate_vector(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 tensor")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")


def _validate_metric(metric: Tensor, dimension: int, name: str) -> Tensor:
    if not isinstance(metric, Tensor) or metric.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not metric.is_floating_point() or not torch.isfinite(metric).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    symmetric = 0.5 * (metric + metric.transpose(-1, -2))
    tolerance = 100 * torch.finfo(metric.dtype).eps * max(1, dimension)
    if not torch.allclose(metric, symmetric, rtol=0.0, atol=float(tolerance)):
        raise ValueError(f"{name} must be symmetric")
    return symmetric


def functional_trust_region_step(
    gradient: Tensor,
    metric: Tensor,
    *,
    trust_budget: float,
    regularizer_metric: Tensor | None = None,
    regularizer_weight: float = 0.0,
) -> FunctionalTrustRegionStep:
    """Solve a local linear objective under a quadratic functional budget.

    The constraint is ``0.5 * step.T @ metric @ step <= trust_budget``.
    A regularizer is permitted only as a metric tensor that is transported with
    the chart. Adding a coordinate identity matrix is intentionally not part of
    this API because it would generally destroy reparameterization equivariance.
    """

    _validate_vector(gradient, "gradient")
    if not math.isfinite(trust_budget) or trust_budget <= 0:
        raise ValueError("trust_budget must be finite and positive")
    if not math.isfinite(regularizer_weight) or regularizer_weight < 0:
        raise ValueError("regularizer_weight must be finite and nonnegative")

    dimension = gradient.numel()
    metric = _validate_metric(metric, dimension, "metric")
    solve_metric = metric
    if regularizer_metric is not None:
        regularizer_metric = _validate_metric(
            regularizer_metric, dimension, "regularizer_metric"
        )
        solve_metric = metric + regularizer_weight * regularizer_metric
    elif regularizer_weight != 0.0:
        raise ValueError("regularizer_weight requires regularizer_metric")

    eigenvalues = torch.linalg.eigvalsh(solve_metric)
    if float(eigenvalues.min()) <= 0.0:
        raise ValueError("the solve metric must be positive definite")
    condition_number = float((eigenvalues.max() / eigenvalues.min()).detach().cpu())

    factor = torch.linalg.cholesky(solve_metric)
    direction = torch.cholesky_solve(
        gradient.unsqueeze(-1), factor
    ).squeeze(-1)
    directional_cost = torch.dot(direction, metric @ direction)
    if not torch.isfinite(directional_cost) or float(directional_cost) <= 0.0:
        raise ValueError("the functional metric assigns no positive cost to the step")

    scale_tensor = torch.sqrt(
        torch.as_tensor(
            2.0 * trust_budget,
            dtype=gradient.dtype,
            device=gradient.device,
        )
        / directional_cost
    )
    step = scale_tensor * direction
    trust_cost = 0.5 * torch.dot(step, metric @ step)
    predicted_gain = torch.dot(gradient, step)
    return FunctionalTrustRegionStep(
        step=step,
        direction=direction,
        scale=float(scale_tensor.detach().cpu()),
        predicted_gain=float(predicted_gain.detach().cpu()),
        trust_cost=float(trust_cost.detach().cpu()),
        metric_condition_number=condition_number,
    )


def euclidean_rms_step(gradient: Tensor, *, rms_step: float) -> Tensor:
    _validate_vector(gradient, "gradient")
    if not math.isfinite(rms_step) or rms_step <= 0:
        raise ValueError("rms_step must be finite and positive")
    rms = torch.sqrt(torch.mean(gradient.square()))
    if float(rms) <= 0.0:
        raise ValueError("gradient must be nonzero")
    return gradient * (rms_step / rms)


def _validate_chart_matrix(matrix: Tensor, dimension: int) -> Tensor:
    if not isinstance(matrix, Tensor) or matrix.shape != (dimension, dimension):
        raise ValueError(f"chart matrix must have shape ({dimension}, {dimension})")
    if not matrix.is_floating_point() or not torch.isfinite(matrix).all():
        raise ValueError("chart matrix must contain finite floating-point values")
    if int(torch.linalg.matrix_rank(matrix)) != dimension:
        raise ValueError("chart matrix must be full rank")
    return matrix


def gradient_to_linear_chart(gradient_native: Tensor, matrix: Tensor) -> Tensor:
    """Transport a covector under ``u = A z + b``: ``g_u = A^-T g_z``."""

    _validate_vector(gradient_native, "gradient_native")
    matrix = _validate_chart_matrix(matrix, gradient_native.numel()).to(
        dtype=gradient_native.dtype, device=gradient_native.device
    )
    return torch.linalg.solve(matrix.transpose(-1, -2), gradient_native)


def metric_to_linear_chart(metric_native: Tensor, matrix: Tensor) -> Tensor:
    """Transport a metric under ``u = A z + b``: ``G_u = A^-T G_z A^-1``."""

    if not isinstance(metric_native, Tensor) or metric_native.ndim != 2:
        raise ValueError("metric_native must be a rank-2 tensor")
    dimension = int(metric_native.shape[0])
    metric_native = _validate_metric(metric_native, dimension, "metric_native")
    matrix = _validate_chart_matrix(matrix, dimension).to(
        dtype=metric_native.dtype, device=metric_native.device
    )
    inverse = torch.linalg.solve(
        matrix,
        torch.eye(dimension, dtype=matrix.dtype, device=matrix.device),
    )
    return inverse.transpose(-1, -2) @ metric_native @ inverse


def vector_from_linear_chart(vector_charted: Tensor, matrix: Tensor) -> Tensor:
    """Transport a tangent vector back to native coordinates."""

    _validate_vector(vector_charted, "vector_charted")
    matrix = _validate_chart_matrix(matrix, vector_charted.numel()).to(
        dtype=vector_charted.dtype, device=vector_charted.device
    )
    return torch.linalg.solve(matrix, vector_charted)


def relative_l2_error(reference: Tensor, candidate: Tensor) -> float:
    _validate_vector(reference, "reference")
    _validate_vector(candidate, "candidate")
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have identical shape")
    denominator = torch.linalg.vector_norm(reference)
    if float(denominator) <= 0.0:
        raise ValueError("reference must be nonzero")
    return float(
        (torch.linalg.vector_norm(candidate - reference) / denominator)
        .detach()
        .cpu()
    )
