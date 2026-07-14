from __future__ import annotations

import math

import torch

from research.coordinate_invariance.switch_c2_geometry import (
    chart_euclidean_update,
    consequential_basis,
    fit_diagonal_whitening_precision,
    normalize_to_predicted_gain,
    prefix_geometry,
    projected_diagonal_metric,
    regularize_with_whitening,
    spearman_correlation,
    top_risk_recall,
)


def test_consequential_basis_is_deterministic_orthonormal_and_gradient_aligned() -> None:
    gradient = torch.tensor([1.0, -2.0, 3.0, 0.5, -0.25])
    first = consequential_basis(gradient, dimension=4, seed=71551)
    second = consequential_basis(gradient, dimension=4, seed=71551)
    assert torch.equal(first, second)
    assert torch.allclose(first.T @ first, torch.eye(4, dtype=torch.float64), atol=1e-12)
    expected = gradient.to(torch.float64) / torch.linalg.vector_norm(
        gradient.to(torch.float64)
    )
    assert torch.allclose(first[:, 0], expected, atol=1e-12)


def test_prefix_geometry_matches_binary_softmax_fisher_and_score() -> None:
    logits = torch.zeros((2, 2), dtype=torch.float64)
    jacobian = torch.tensor(
        [
            [[1.0, 0.0], [-1.0, 0.0]],
            [[0.0, 2.0], [0.0, -2.0]],
        ],
        dtype=torch.float64,
    )
    geometry = prefix_geometry(
        logits,
        jacobian,
        target_ids=torch.tensor([0, 1]),
        objective_horizon=2,
    )
    assert torch.allclose(
        geometry.cumulative_metrics[0],
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
    )
    assert torch.allclose(
        geometry.cumulative_metrics[1],
        torch.tensor([[1.0, 0.0], [0.0, 4.0]], dtype=torch.float64),
    )
    assert torch.allclose(
        geometry.objective_gradient,
        torch.tensor([0.5, -1.0], dtype=torch.float64),
    )


def test_whitening_ridge_and_gain_normalization_remain_tensorial() -> None:
    states = torch.tensor(
        [[1.0, 2.0, -1.0], [2.0, 4.0, 0.0], [4.0, 8.0, 1.0]],
        dtype=torch.float64,
    )
    precision, floor = fit_diagonal_whitening_precision(
        states, variance_floor_fraction_of_median=0.001
    )
    assert floor > 0.0
    basis = torch.eye(3, 2, dtype=torch.float64)
    whitening = projected_diagonal_metric(basis, precision)
    metric = torch.tensor([[2.0, 0.2], [0.2, 0.5]], dtype=torch.float64)
    regularized, ridge = regularize_with_whitening(
        metric, whitening, relative_generalized_ridge=0.01
    )
    assert ridge > 0.0
    assert float(torch.linalg.eigvalsh(regularized).min()) > 0.0
    gradient = torch.tensor([0.7, -0.3], dtype=torch.float64)
    direction = torch.linalg.solve(regularized, gradient)
    step = normalize_to_predicted_gain(direction, gradient, predicted_gain=0.003)
    assert math.isclose(float(torch.dot(gradient, step)), 0.003, abs_tol=1e-12)


def test_anisotropic_chart_changes_euclidean_direction_after_scalar_retuning() -> None:
    gradient = torch.tensor([1.0, 2.0], dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    anisotropic = torch.tensor([[4.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    native = chart_euclidean_update(gradient, identity, predicted_gain=0.01)
    charted = chart_euclidean_update(gradient, anisotropic, predicted_gain=0.01)
    assert math.isclose(float(torch.dot(gradient, native)), 0.01, abs_tol=1e-12)
    assert math.isclose(float(torch.dot(gradient, charted)), 0.01, abs_tol=1e-12)
    assert not torch.allclose(native, charted)


def test_rank_statistics_handle_ties_and_top_fraction() -> None:
    predicted = [1.0, 1.0, 2.0, 3.0, 4.0]
    actual = [0.0, 1.0, 2.0, 3.0, 4.0]
    correlation = spearman_correlation(predicted, actual)
    assert 0.9 < correlation < 1.0
    assert top_risk_recall(predicted, actual, fraction=0.4) == 1.0
