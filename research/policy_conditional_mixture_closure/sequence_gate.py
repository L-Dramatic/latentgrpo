"""Exact counterexample separating one-step and sequence-level closure."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping


Distribution = Mapping[str, float]


def _validate_distribution(distribution: Distribution) -> None:
    if not distribution:
        raise ValueError("distribution must be non-empty")
    if any(probability < 0.0 for probability in distribution.values()):
        raise ValueError("probabilities must be non-negative")
    if not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-12):
        raise ValueError("probabilities must sum to one")


def total_variation(left: Distribution, right: Distribution) -> float:
    _validate_distribution(left)
    _validate_distribution(right)
    support = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(outcome, 0.0) - right.get(outcome, 0.0))
        for outcome in support
    )


def categorical_kl(left: Distribution, right: Distribution) -> float:
    _validate_distribution(left)
    _validate_distribution(right)
    divergence = 0.0
    for outcome, probability in left.items():
        if probability == 0.0:
            continue
        reference = right.get(outcome, 0.0)
        if reference == 0.0:
            return math.inf
        divergence += probability * math.log(probability / reference)
    return divergence


def first_token_marginal(distribution: Distribution) -> dict[str, float]:
    _validate_distribution(distribution)
    marginal: dict[str, float] = {}
    for sequence, probability in distribution.items():
        if not sequence:
            raise ValueError("sequence outcomes must be non-empty")
        marginal[sequence[0]] = marginal.get(sequence[0], 0.0) + probability
    return marginal


@dataclass(frozen=True)
class SequenceClosureGate:
    initial_one_step_kl_nats: float
    static_student_sequence_tv: float
    static_student_cross_branch_mass: float
    posterior_updated_student_sequence_tv: float
    one_step_closure_implies_sequence_closure: bool


def run_sequence_closure_gate() -> SequenceClosureGate:
    """Evaluate a two-branch deterministic autoregressive counterexample.

    A hidden branch is selected once. Branch A emits 00 and branch B emits 11.
    The correct sequence mixture therefore has no 01 or 10 mass. A student can
    exactly match the initial next-token mixture while independently reusing
    static weights at the second step, creating 50% impossible cross-branch
    sequences. Updating branch weights from the observed prefix removes the
    error, but that requirement is absent from a one-step initial-state loss.
    """

    teacher = {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5}
    static_student = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
    posterior_updated_student = dict(teacher)

    initial_kl = categorical_kl(
        first_token_marginal(teacher), first_token_marginal(static_student)
    )
    sequence_tv = total_variation(teacher, static_student)
    posterior_tv = total_variation(teacher, posterior_updated_student)
    cross_branch_mass = static_student["01"] + static_student["10"]
    return SequenceClosureGate(
        initial_one_step_kl_nats=initial_kl,
        static_student_sequence_tv=sequence_tv,
        static_student_cross_branch_mass=cross_branch_mass,
        posterior_updated_student_sequence_tv=posterior_tv,
        one_step_closure_implies_sequence_closure=(
            initial_kl > 1e-12 or sequence_tv < 1e-12
        ),
    )


if __name__ == "__main__":
    print(json.dumps(asdict(run_sequence_closure_gate()), indent=2, sort_keys=True))

