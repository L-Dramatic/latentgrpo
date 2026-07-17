"""Exact identity and identifiability checks for OMPI-R.

The gate works on a finite candidate set. Rows index sampled private latent
paths and columns index distinct visible responses.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Sequence


Matrix = Sequence[Sequence[float]]


def _validate_matrix(log_likelihoods: Matrix) -> tuple[int, int]:
    if not log_likelihoods or not log_likelihoods[0]:
        raise ValueError("log_likelihoods must be a non-empty matrix")
    columns = len(log_likelihoods[0])
    if any(len(row) != columns for row in log_likelihoods):
        raise ValueError("log_likelihoods must be rectangular")
    return len(log_likelihoods), columns


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    normalizer = _logsumexp(logits)
    return tuple(math.exp(logit - normalizer) for logit in logits)


def marginal_logits(log_likelihoods: Matrix) -> tuple[float, ...]:
    """Return log mean likelihood over private paths for every response."""

    rows, columns = _validate_matrix(log_likelihoods)
    log_rows = math.log(rows)
    return tuple(
        _logsumexp([log_likelihoods[row][column] for row in range(rows)])
        - log_rows
        for column in range(columns)
    )


def responsibilities(log_likelihoods: Matrix) -> tuple[tuple[float, ...], ...]:
    """Return response-to-path responsibilities with shape [response, path]."""

    rows, columns = _validate_matrix(log_likelihoods)
    return tuple(
        softmax([log_likelihoods[row][column] for row in range(rows)])
        for column in range(columns)
    )


def reward_tilted_target(
    old_candidate_logits: Sequence[float],
    rewards: Sequence[float],
    *,
    beta: float,
) -> tuple[float, ...]:
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if len(old_candidate_logits) != len(rewards):
        raise ValueError("old_candidate_logits and rewards must have equal length")
    return softmax(
        [logit + reward / beta for logit, reward in zip(old_candidate_logits, rewards)]
    )


def tpo_candidate_loss(
    candidate_logits: Sequence[float],
    old_candidate_logits: Sequence[float],
    rewards: Sequence[float],
    *,
    beta: float,
) -> float:
    """TPO cross-entropy on an arbitrary finite candidate-logit simplex."""

    target = reward_tilted_target(old_candidate_logits, rewards, beta=beta)
    log_normalizer = _logsumexp(candidate_logits)
    return -sum(
        target_value * (logit - log_normalizer)
        for target_value, logit in zip(target, candidate_logits)
    )


def ompi_loss(
    log_likelihoods: Matrix,
    old_log_likelihoods: Matrix,
    rewards: Sequence[float],
    *,
    beta: float,
) -> float:
    """OMPI-R loss from the report, written without implementation shortcuts."""

    current_marginal = marginal_logits(log_likelihoods)
    old_marginal = marginal_logits(old_log_likelihoods)
    target = reward_tilted_target(old_marginal, rewards, beta=beta)
    current_probabilities = softmax(current_marginal)
    return -sum(
        target_value * math.log(probability)
        for target_value, probability in zip(target, current_probabilities)
    )


def ompi_likelihood_gradient(
    log_likelihoods: Matrix,
    old_log_likelihoods: Matrix,
    rewards: Sequence[float],
    *,
    beta: float,
) -> tuple[tuple[float, ...], ...]:
    """Gradient of OMPI-R with respect to the all-pairs likelihood matrix."""

    rows, columns = _validate_matrix(log_likelihoods)
    probabilities = softmax(marginal_logits(log_likelihoods))
    target = reward_tilted_target(
        marginal_logits(old_log_likelihoods), rewards, beta=beta
    )
    response_responsibilities = responsibilities(log_likelihoods)
    return tuple(
        tuple(
            (probabilities[column] - target[column])
            * response_responsibilities[column][row]
            for column in range(columns)
        )
        for row in range(rows)
    )


def tpo_on_marginal_likelihood_gradient(
    log_likelihoods: Matrix,
    old_log_likelihoods: Matrix,
    rewards: Sequence[float],
    *,
    beta: float,
) -> tuple[tuple[float, ...], ...]:
    """Chain-rule gradient of TPO after JEPO-style empirical marginalization."""

    rows, columns = _validate_matrix(log_likelihoods)
    candidate_probabilities = softmax(marginal_logits(log_likelihoods))
    target_probabilities = reward_tilted_target(
        marginal_logits(old_log_likelihoods), rewards, beta=beta
    )
    gradients = [[0.0 for _ in range(columns)] for _ in range(rows)]
    for column in range(columns):
        column_weights = softmax(
            [log_likelihoods[row][column] for row in range(rows)]
        )
        marginal_gradient = (
            candidate_probabilities[column] - target_probabilities[column]
        )
        for row, weight in enumerate(column_weights):
            gradients[row][column] = marginal_gradient * weight
    return tuple(tuple(row) for row in gradients)


def responsibility_ess(log_likelihoods: Matrix) -> tuple[float, ...]:
    return tuple(
        1.0 / sum(weight * weight for weight in response_weights)
        for response_weights in responsibilities(log_likelihoods)
    )


def normalized_responsibility_entropy(
    log_likelihoods: Matrix,
) -> tuple[float, ...]:
    rows, _ = _validate_matrix(log_likelihoods)
    if rows == 1:
        return tuple(0.0 for _ in responsibilities(log_likelihoods))
    scale = math.log(rows)
    return tuple(
        -sum(weight * math.log(weight) for weight in response_weights) / scale
        for response_weights in responsibilities(log_likelihoods)
    )


def off_source_mass(
    log_likelihoods: Matrix,
    source_paths: Sequence[int],
) -> tuple[float, ...]:
    rows, columns = _validate_matrix(log_likelihoods)
    if len(source_paths) != columns:
        raise ValueError("source_paths must contain one path index per response")
    response_responsibilities = responsibilities(log_likelihoods)
    masses: list[float] = []
    for column, source_path in enumerate(source_paths):
        if not 0 <= source_path < rows:
            raise ValueError("source path index out of range")
        masses.append(1.0 - response_responsibilities[column][source_path])
    return tuple(masses)


def path_candidate_mutual_information(log_likelihoods: Matrix) -> float:
    """Candidate-set path sensitivity under a uniform path intervention.

    This is zero when every path induces the same normalized distribution over
    the visible candidate set, including the latent-irrelevance null that makes
    responsibility ESS maximally large.
    """

    rows, columns = _validate_matrix(log_likelihoods)
    conditionals = [softmax(row) for row in log_likelihoods]
    mixture = [
        sum(conditionals[row][column] for row in range(rows)) / rows
        for column in range(columns)
    ]
    return sum(
        conditional[column]
        * math.log(conditional[column] / mixture[column])
        for conditional in conditionals
        for column in range(columns)
    ) / rows


@dataclass(frozen=True)
class IdentityGateResult:
    loss_absolute_difference: float
    gradient_max_absolute_difference: float
    irrelevance_median_ess: float
    irrelevance_median_normalized_entropy: float
    irrelevance_median_off_source_mass: float
    irrelevance_path_candidate_mi_nats: float
    irrelevance_candidate_probability_l1: float
    report_gate_1a_would_false_pass: bool
    standalone_optimizer_identity_passes: bool


def run_identity_gate() -> IdentityGateResult:
    current = (
        (-0.30, -1.20, -2.10, -1.70),
        (-0.45, -0.95, -2.30, -1.35),
        (-1.25, -0.80, -1.15, -2.00),
        (-1.70, -1.10, -0.75, -0.90),
    )
    old = tuple(tuple(value - 0.07 for value in row) for row in current)
    rewards = (1.0, 0.0, 1.0, -0.5)
    beta = 0.7

    ompi = ompi_loss(current, old, rewards, beta=beta)
    composed = tpo_candidate_loss(
        marginal_logits(current), marginal_logits(old), rewards, beta=beta
    )
    ompi_gradient = ompi_likelihood_gradient(current, old, rewards, beta=beta)
    composed_gradient = tpo_on_marginal_likelihood_gradient(
        current, old, rewards, beta=beta
    )
    gradient_difference = max(
        abs(ompi_gradient[row][column] - composed_gradient[row][column])
        for row in range(len(current))
        for column in range(len(current[0]))
    )

    base = softmax((-0.15, -1.10, -1.75, -2.30))
    irrelevance = tuple(tuple(math.log(value) for value in base) for _ in range(4))
    ess = responsibility_ess(irrelevance)
    entropy = normalized_responsibility_entropy(irrelevance)
    off_source = off_source_mass(irrelevance, (0, 1, 2, 3))
    marginal_probability = softmax(marginal_logits(irrelevance))
    probability_l1 = sum(
        abs(left - right) for left, right in zip(marginal_probability, base)
    )
    path_mi = path_candidate_mutual_information(irrelevance)

    report_gate_false_pass = (
        median(ess) >= 1.50
        and median(entropy) >= 0.25
        and median(off_source) >= 0.25
        and path_mi < 1e-12
        and probability_l1 < 1e-12
    )
    return IdentityGateResult(
        loss_absolute_difference=abs(ompi - composed),
        gradient_max_absolute_difference=gradient_difference,
        irrelevance_median_ess=median(ess),
        irrelevance_median_normalized_entropy=median(entropy),
        irrelevance_median_off_source_mass=median(off_source),
        irrelevance_path_candidate_mi_nats=path_mi,
        irrelevance_candidate_probability_l1=probability_l1,
        report_gate_1a_would_false_pass=report_gate_false_pass,
        standalone_optimizer_identity_passes=(
            abs(ompi - composed) < 1e-12 and gradient_difference < 1e-12
        ),
    )


if __name__ == "__main__":
    print(json.dumps(asdict(run_identity_gate()), indent=2, sort_keys=True))

