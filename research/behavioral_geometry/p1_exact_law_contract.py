"""Exact finite-vocabulary law checks for P1 density and support gates.

The real P1 estimator is Monte Carlo over a large vocabulary.  Before touching
a checkpoint, its full-support and truncated-law semantics must agree with
small-vocabulary enumeration, where the joint distribution and its chain-rule
KL can be computed exactly rather than inferred from rollouts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


LogitRule = Callable[[tuple[int, ...]], Tensor]


@dataclass(frozen=True)
class ExactAutoregressiveLaw:
    """Complete finite-horizon joint law indexed by token tuples."""

    horizon: int
    vocabulary_size: int
    joint_probabilities: dict[tuple[int, ...], Tensor]

    @property
    def total_probability(self) -> Tensor:
        return torch.stack(tuple(self.joint_probabilities.values())).sum()

    @property
    def has_full_joint_support(self) -> bool:
        return all(bool((value > 0).item()) for value in self.joint_probabilities.values())


def full_support_probs_from_logits(logits: Tensor) -> Tensor:
    """Return the exact categorical law used by an untruncated softmax sampler."""

    if logits.ndim != 1 or not logits.is_floating_point():
        raise TypeError("logits must be one floating-point vocabulary vector")
    if logits.numel() < 2 or not torch.isfinite(logits).all():
        raise ValueError("finite logits for at least two vocabulary items are required")
    probs = torch.softmax(logits.to(torch.float64), dim=-1)
    if not torch.all(probs > 0):
        raise FloatingPointError("finite full-support softmax underflowed to zero")
    return probs


def source_visible_top_p_probs(logits: Tensor, top_p: float) -> Tensor:
    """Match pinned ``top_p_normalize_probs_torch`` support semantics.

    This is the visible-decoding helper's ``> top_p`` exclusion rule.  It is
    intentionally separate from the latent sampler's strict ``< top_p`` plus
    forced-top-k rule, which is checked by the sampler replay tests.
    """

    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    probs = full_support_probs_from_logits(logits)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[(cumulative - sorted_probs) > top_p] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()
    return torch.zeros_like(sorted_probs).scatter_(0, sorted_ids, sorted_probs)


def enumerate_autoregressive_law(
    rule: LogitRule, *, horizon: int, vocabulary_size: int
) -> ExactAutoregressiveLaw:
    """Enumerate a full-support finite autoregressive joint distribution."""

    if horizon < 1 or vocabulary_size < 2:
        raise ValueError("horizon and vocabulary_size must both be at least two/one")
    prefixes: dict[tuple[int, ...], Tensor] = {(): torch.tensor(1.0, dtype=torch.float64)}
    for _ in range(horizon):
        next_prefixes: dict[tuple[int, ...], Tensor] = {}
        for prefix, prefix_prob in prefixes.items():
            probs = full_support_probs_from_logits(rule(prefix))
            if probs.numel() != vocabulary_size:
                raise ValueError("rule returned an unexpected vocabulary size")
            for token_id in range(vocabulary_size):
                next_prefixes[prefix + (token_id,)] = prefix_prob * probs[token_id]
        prefixes = next_prefixes
    return ExactAutoregressiveLaw(horizon, vocabulary_size, prefixes)


def exact_joint_kl(left: ExactAutoregressiveLaw, right: ExactAutoregressiveLaw) -> Tensor:
    """Compute KL(left || right), returning infinity on a support violation."""

    if (left.horizon, left.vocabulary_size) != (right.horizon, right.vocabulary_size):
        raise ValueError("exact laws must share horizon and vocabulary")
    result = torch.tensor(0.0, dtype=torch.float64)
    for sequence, left_prob in left.joint_probabilities.items():
        right_prob = right.joint_probabilities.get(sequence, torch.tensor(0.0, dtype=torch.float64))
        if left_prob > 0 and right_prob <= 0:
            return torch.tensor(float("inf"), dtype=torch.float64)
        if left_prob > 0:
            result = result + left_prob * (left_prob.log() - right_prob.log())
    return result


def exact_chain_rule_kl(
    left_rule: LogitRule,
    right_rule: LogitRule,
    *,
    horizon: int,
    vocabulary_size: int,
) -> Tensor:
    """Enumerate the expected categorical chain-rule KL under ``left_rule``."""

    if horizon < 1 or vocabulary_size < 2:
        raise ValueError("horizon and vocabulary_size must both be at least two/one")
    prefixes: dict[tuple[int, ...], Tensor] = {(): torch.tensor(1.0, dtype=torch.float64)}
    total = torch.tensor(0.0, dtype=torch.float64)
    for _ in range(horizon):
        next_prefixes: dict[tuple[int, ...], Tensor] = {}
        for prefix, prefix_prob in prefixes.items():
            left_probs = full_support_probs_from_logits(left_rule(prefix))
            right_probs = full_support_probs_from_logits(right_rule(prefix))
            if left_probs.numel() != vocabulary_size or right_probs.numel() != vocabulary_size:
                raise ValueError("rule returned an unexpected vocabulary size")
            total = total + prefix_prob * torch.sum(
                left_probs * (left_probs.log() - right_probs.log())
            )
            for token_id in range(vocabulary_size):
                next_prefixes[prefix + (token_id,)] = prefix_prob * left_probs[token_id]
        prefixes = next_prefixes
    return total
