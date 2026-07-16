"""Exact one-step audit for marginal-preserving correlated GRPO groups."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointOutcome:
    first: int
    second: int
    probability: float


@dataclass(frozen=True)
class EstimatorMoments:
    expectation: float
    variance: float


@dataclass(frozen=True)
class CoupledGroupGate:
    success_probability: float
    true_gradient: float
    iid_informative_probability: float
    antithetic_informative_probability: float
    iid_raw: EstimatorMoments
    antithetic_raw: EstimatorMoments
    iid_leave_one_out: EstimatorMoments
    antithetic_leave_one_out: EstimatorMoments
    first_marginal_antithetic: float
    second_marginal_antithetic: float

    @property
    def antithetic_leave_one_out_bias(self) -> float:
        return self.antithetic_leave_one_out.expectation - self.true_gradient

    @property
    def antithetic_leave_one_out_relative_bias(self) -> float:
        return self.antithetic_leave_one_out_bias / self.true_gradient

    @property
    def raw_variance_ratio(self) -> float:
        return self.antithetic_raw.variance / self.iid_raw.variance


def _validate_probability(probability: float) -> None:
    if not 0.0 < probability < 1.0:
        raise ValueError("success_probability must lie strictly between zero and one")


def iid_joint(success_probability: float) -> tuple[JointOutcome, ...]:
    """Joint law of two independent Bernoulli policy samples."""

    _validate_probability(success_probability)
    p = success_probability
    return (
        JointOutcome(0, 0, (1.0 - p) ** 2),
        JointOutcome(0, 1, (1.0 - p) * p),
        JointOutcome(1, 0, p * (1.0 - p)),
        JointOutcome(1, 1, p**2),
    )


def antithetic_joint(success_probability: float) -> tuple[JointOutcome, ...]:
    """Joint law induced by U and 1-U through Bernoulli inverse CDFs."""

    _validate_probability(success_probability)
    p = success_probability
    boundaries = sorted({0.0, p, 1.0 - p, 1.0})
    probabilities: dict[tuple[int, int], float] = {}
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        outcome = (int(midpoint < p), int(1.0 - midpoint < p))
        probabilities[outcome] = probabilities.get(outcome, 0.0) + right - left
    return tuple(
        JointOutcome(first, second, probability)
        for (first, second), probability in sorted(probabilities.items())
    )


def _moments(
    law: tuple[JointOutcome, ...],
    success_probability: float,
    *,
    leave_one_out: bool,
) -> EstimatorMoments:
    values: list[tuple[float, float]] = []
    p = success_probability
    for outcome in law:
        rewards = (float(outcome.first), float(outcome.second))
        scores = (outcome.first - p, outcome.second - p)
        if leave_one_out:
            advantages = (rewards[0] - rewards[1], rewards[1] - rewards[0])
        else:
            advantages = rewards
        estimate = sum(
            advantage * score for advantage, score in zip(advantages, scores)
        ) / 2.0
        values.append((outcome.probability, estimate))
    expectation = sum(probability * value for probability, value in values)
    second_moment = sum(probability * value**2 for probability, value in values)
    return EstimatorMoments(expectation, second_moment - expectation**2)


def _informative_probability(law: tuple[JointOutcome, ...]) -> float:
    return sum(
        outcome.probability
        for outcome in law
        if outcome.first != outcome.second
    )


def _marginal(
    law: tuple[JointOutcome, ...],
    *,
    member: int,
) -> float:
    return sum(
        outcome.probability
        for outcome in law
        if (outcome.first, outcome.second)[member] == 1
    )


def evaluate_coupled_group_gate(success_probability: float) -> CoupledGroupGate:
    """Evaluate coverage, bias, and variance for the exact two-member gate."""

    iid = iid_joint(success_probability)
    antithetic = antithetic_joint(success_probability)
    return CoupledGroupGate(
        success_probability=success_probability,
        true_gradient=success_probability * (1.0 - success_probability),
        iid_informative_probability=_informative_probability(iid),
        antithetic_informative_probability=_informative_probability(antithetic),
        iid_raw=_moments(iid, success_probability, leave_one_out=False),
        antithetic_raw=_moments(
            antithetic, success_probability, leave_one_out=False
        ),
        iid_leave_one_out=_moments(iid, success_probability, leave_one_out=True),
        antithetic_leave_one_out=_moments(
            antithetic, success_probability, leave_one_out=True
        ),
        first_marginal_antithetic=_marginal(antithetic, member=0),
        second_marginal_antithetic=_marginal(antithetic, member=1),
    )
