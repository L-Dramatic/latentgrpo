import pytest
import torch

from research.coordinate_invariance.fctr import (
    euclidean_rms_step,
    functional_trust_region_step,
    gradient_to_linear_chart,
    metric_to_linear_chart,
    relative_l2_error,
    vector_from_linear_chart,
)


def random_problem(seed=7, dimension=6):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(
        dimension, dimension, generator=generator, dtype=torch.float64
    )
    metric = raw.T @ raw + 0.5 * torch.eye(dimension, dtype=torch.float64)
    gradient = torch.randn(dimension, generator=generator, dtype=torch.float64)
    chart = torch.randn(
        dimension, dimension, generator=generator, dtype=torch.float64
    )
    chart = chart + 2.0 * torch.eye(dimension, dtype=torch.float64)
    return gradient, metric, chart


def test_fctr_step_uses_exact_quadratic_budget():
    gradient, metric, _ = random_problem()
    result = functional_trust_region_step(
        gradient, metric, trust_budget=0.03
    )
    assert result.trust_cost == pytest.approx(0.03, abs=1e-12)
    assert result.predicted_gain > 0.0


def test_fctr_step_is_equivariant_under_linear_chart():
    gradient, metric, chart = random_problem(seed=11)
    native = functional_trust_region_step(
        gradient, metric, trust_budget=0.02
    )
    charted = functional_trust_region_step(
        gradient_to_linear_chart(gradient, chart),
        metric_to_linear_chart(metric, chart),
        trust_budget=0.02,
    )
    transported = vector_from_linear_chart(charted.step, chart)
    assert relative_l2_error(native.step, transported) < 1e-11
    assert charted.predicted_gain == pytest.approx(
        native.predicted_gain, abs=1e-11
    )


def test_covariant_regularizer_preserves_equivariance():
    gradient, metric, chart = random_problem(seed=13)
    regularizer = torch.diag(
        torch.linspace(0.5, 1.5, gradient.numel(), dtype=torch.float64)
    )
    native = functional_trust_region_step(
        gradient,
        metric,
        trust_budget=0.01,
        regularizer_metric=regularizer,
        regularizer_weight=0.2,
    )
    charted = functional_trust_region_step(
        gradient_to_linear_chart(gradient, chart),
        metric_to_linear_chart(metric, chart),
        trust_budget=0.01,
        regularizer_metric=metric_to_linear_chart(regularizer, chart),
        regularizer_weight=0.2,
    )
    transported = vector_from_linear_chart(charted.step, chart)
    assert relative_l2_error(native.step, transported) < 1e-11


def test_euclidean_step_is_not_anisotropic_chart_invariant():
    gradient = torch.tensor([1.0, 1.0], dtype=torch.float64)
    chart = torch.diag(torch.tensor([4.0, 0.25], dtype=torch.float64))
    native = euclidean_rms_step(gradient, rms_step=0.1)
    charted = euclidean_rms_step(
        gradient_to_linear_chart(gradient, chart), rms_step=0.1
    )
    transported = vector_from_linear_chart(charted, chart)
    assert relative_l2_error(native, transported) > 1.0


def test_solver_rejects_coordinate_identity_damping_shortcut():
    gradient, metric, _ = random_problem()
    with pytest.raises(ValueError, match="requires regularizer_metric"):
        functional_trust_region_step(
            gradient,
            metric,
            trust_budget=0.01,
            regularizer_weight=0.1,
        )
