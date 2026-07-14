from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from .densities import (
    naive_selected_gumbel_log_density,
    sample_topk_concrete,
    selected_gumbel_log_density,
    topk_concrete_log_density,
)


@dataclass(frozen=True)
class ToyGateConfig:
    experiment_name: str
    vocabulary_size: int
    top_k: int
    sample_count: int
    score_sample_count: int
    scenario_seeds: tuple[int, ...]
    temperature: float
    gumbel_scale: float
    logit_scale: float
    policy_drift_scale: float
    reward_nonlinearity: float
    clip_epsilon: float
    thresholds: dict[str, float]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ToyGateConfig":
        config = cls(
            experiment_name=str(raw["experiment_name"]),
            vocabulary_size=int(raw["vocabulary_size"]),
            top_k=int(raw["top_k"]),
            sample_count=int(raw["sample_count"]),
            score_sample_count=int(raw["score_sample_count"]),
            scenario_seeds=tuple(int(value) for value in raw["scenario_seeds"]),
            temperature=float(raw["temperature"]),
            gumbel_scale=float(raw["gumbel_scale"]),
            logit_scale=float(raw["logit_scale"]),
            policy_drift_scale=float(raw["policy_drift_scale"]),
            reward_nonlinearity=float(raw["reward_nonlinearity"]),
            clip_epsilon=float(raw["clip_epsilon"]),
            thresholds={key: float(value) for key, value in raw["thresholds"].items()},
        )
        if config.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least two")
        if not 1 <= config.top_k <= config.vocabulary_size:
            raise ValueError("top_k must fit within vocabulary_size")
        if config.sample_count < 1 or config.score_sample_count < 1:
            raise ValueError("sample counts must be positive")
        if not config.scenario_seeds:
            raise ValueError("at least one scenario seed is required")
        return config


def _ppo_clip_decision(log_ratio: Tensor, advantages: Tensor, epsilon: float) -> Tensor:
    upper = math.log1p(epsilon)
    lower = math.log1p(-epsilon)
    return torch.where(advantages >= 0, log_ratio > upper, log_ratio < lower)


def _mean_score_norm(
    logits: Tensor,
    log_density: Callable[[Tensor], Tensor],
) -> float:
    differentiable_logits = logits.detach().clone().requires_grad_(True)
    mean_log_density = log_density(differentiable_logits).mean()
    score = torch.autograd.grad(mean_log_density, differentiable_logits)[0]
    return float(score.norm())


def _cosine_similarity(first: Tensor, second: Tensor) -> float:
    denominator = first.norm() * second.norm()
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(first, second) else 0.0
    return float(torch.dot(first, second) / denominator)


def _reference_density_error(config: ToyGateConfig) -> dict[str, float]:
    dtype = torch.float64
    generator = torch.Generator(device="cpu")
    generator.manual_seed(91001)
    logits = torch.linspace(-0.9, 0.8, config.vocabulary_size, dtype=dtype)

    categorical_samples = sample_topk_concrete(
        logits,
        top_k=1,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
        sample_shape=(1024,),
        generator=generator,
    )
    categorical_density = topk_concrete_log_density(
        categorical_samples.weights,
        categorical_samples.ordered_indices,
        logits,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
    )
    categorical_reference = torch.log_softmax(
        logits / config.gumbel_scale, dim=-1
    ).gather(0, categorical_samples.ordered_indices.squeeze(-1))
    categorical_error = float(
        (categorical_density - categorical_reference).abs().max()
    )

    concrete_samples = sample_topk_concrete(
        logits,
        top_k=config.vocabulary_size,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
        sample_shape=(1024,),
        generator=generator,
    )
    ordered_density = topk_concrete_log_density(
        concrete_samples.weights,
        concrete_samples.ordered_indices,
        logits,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
    )
    category_weights = torch.zeros_like(concrete_samples.weights).scatter(
        -1, concrete_samples.ordered_indices, concrete_samples.weights
    )
    reference = torch.distributions.RelaxedOneHotCategorical(
        temperature=torch.tensor(
            config.temperature / config.gumbel_scale, dtype=dtype
        ),
        logits=logits / config.gumbel_scale,
    ).log_prob(category_weights)
    concrete_error = float((ordered_density - reference).abs().max())
    return {
        "k1_categorical_max_abs_error": categorical_error,
        "kv_concrete_max_abs_error": concrete_error,
        "max_abs_error": max(categorical_error, concrete_error),
    }


def _shift_invariance_error(config: ToyGateConfig) -> float:
    dtype = torch.float64
    generator = torch.Generator(device="cpu")
    generator.manual_seed(91002)
    logits = torch.randn(
        config.vocabulary_size, generator=generator, dtype=dtype
    ) * config.logit_scale
    samples = sample_topk_concrete(
        logits,
        top_k=config.top_k,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
        sample_shape=(2048,),
        generator=generator,
    )
    baseline = topk_concrete_log_density(
        samples.weights,
        samples.ordered_indices,
        logits,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
    )
    errors = []
    for offset in (-100.0, -3.25, 2.75, 100.0):
        shifted = topk_concrete_log_density(
            samples.weights,
            samples.ordered_indices,
            logits + offset,
            temperature=config.temperature,
            gumbel_scale=config.gumbel_scale,
        )
        errors.append(float((shifted - baseline).abs().max()))
    return max(errors)


def _policy_gradient(
    logits: Tensor,
    log_density: Callable[[Tensor], Tensor],
    advantages: Tensor,
) -> Tensor:
    differentiable_logits = logits.detach().clone().requires_grad_(True)
    objective = (advantages * log_density(differentiable_logits)).mean()
    return torch.autograd.grad(objective, differentiable_logits)[0].detach()


def _run_scenario(config: ToyGateConfig, seed: int) -> dict[str, float]:
    dtype = torch.float64
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    old_logits = torch.randn(
        config.vocabulary_size, generator=generator, dtype=dtype
    ) * config.logit_scale
    old_logits = old_logits - old_logits.mean()
    drift = torch.randn(
        config.vocabulary_size, generator=generator, dtype=dtype
    ) * config.policy_drift_scale
    drift = drift - drift.mean()
    current_logits = old_logits + drift
    utility = torch.randn(
        config.vocabulary_size, generator=generator, dtype=dtype
    )
    curvature = torch.randn(
        config.vocabulary_size, generator=generator, dtype=dtype
    )

    samples = sample_topk_concrete(
        old_logits,
        top_k=config.top_k,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
        sample_shape=(config.sample_count,),
        generator=generator,
    )
    selected_utility = torch.gather(
        utility.expand(config.sample_count, -1), -1, samples.ordered_indices
    )
    selected_curvature = torch.gather(
        curvature.expand(config.sample_count, -1), -1, samples.ordered_indices
    )
    linear_reward = (samples.weights * selected_utility).sum(dim=-1)
    nonlinear_reward = (samples.weights * selected_curvature).sum(dim=-1).square()
    rewards = linear_reward + config.reward_nonlinearity * nonlinear_reward
    advantages = rewards - rewards.mean()

    exact_action_old = topk_concrete_log_density(
        samples.weights,
        samples.ordered_indices,
        old_logits,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
    )
    exact_action_current = topk_concrete_log_density(
        samples.weights,
        samples.ordered_indices,
        current_logits,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
    )
    exact_action_log_ratio = exact_action_current - exact_action_old

    exact_auxiliary_old = selected_gumbel_log_density(
        samples.selected_scores,
        samples.ordered_indices,
        old_logits,
        gumbel_scale=config.gumbel_scale,
    )
    exact_auxiliary_current = selected_gumbel_log_density(
        samples.selected_scores,
        samples.ordered_indices,
        current_logits,
        gumbel_scale=config.gumbel_scale,
    )
    exact_auxiliary_log_ratio = exact_auxiliary_current - exact_auxiliary_old

    naive_sum_old = naive_selected_gumbel_log_density(
        samples.selected_scores,
        samples.ordered_indices,
        old_logits,
        gumbel_scale=config.gumbel_scale,
        reduction="sum",
    )
    naive_sum_current = naive_selected_gumbel_log_density(
        samples.selected_scores,
        samples.ordered_indices,
        current_logits,
        gumbel_scale=config.gumbel_scale,
        reduction="sum",
    )
    naive_sum_log_ratio = naive_sum_current - naive_sum_old
    official_mean_log_ratio = naive_sum_log_ratio / config.top_k

    exact_clip = _ppo_clip_decision(
        exact_action_log_ratio, advantages, config.clip_epsilon
    )
    official_clip = _ppo_clip_decision(
        official_mean_log_ratio, advantages, config.clip_epsilon
    )
    selection_correction = exact_auxiliary_log_ratio - naive_sum_log_ratio

    exact_gradient = _policy_gradient(
        old_logits,
        lambda value: topk_concrete_log_density(
            samples.weights,
            samples.ordered_indices,
            value,
            temperature=config.temperature,
            gumbel_scale=config.gumbel_scale,
        ),
        advantages,
    )
    official_gradient = _policy_gradient(
        old_logits,
        lambda value: naive_selected_gumbel_log_density(
            samples.selected_scores,
            samples.ordered_indices,
            value,
            gumbel_scale=config.gumbel_scale,
            reduction="mean",
        ),
        advantages,
    )

    score_samples = sample_topk_concrete(
        old_logits,
        top_k=config.top_k,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
        sample_shape=(config.score_sample_count,),
        generator=generator,
    )
    exact_action_score_norm = _mean_score_norm(
        old_logits,
        lambda value: topk_concrete_log_density(
            score_samples.weights,
            score_samples.ordered_indices,
            value,
            temperature=config.temperature,
            gumbel_scale=config.gumbel_scale,
        ),
    )
    exact_auxiliary_score_norm = _mean_score_norm(
        old_logits,
        lambda value: selected_gumbel_log_density(
            score_samples.selected_scores,
            score_samples.ordered_indices,
            value,
            gumbel_scale=config.gumbel_scale,
        ),
    )
    naive_sum_score_norm = _mean_score_norm(
        old_logits,
        lambda value: naive_selected_gumbel_log_density(
            score_samples.selected_scores,
            score_samples.ordered_indices,
            value,
            gumbel_scale=config.gumbel_scale,
            reduction="sum",
        ),
    )

    return {
        "exact_action_ratio_mean": float(torch.exp(exact_action_log_ratio).mean()),
        "exact_auxiliary_ratio_mean": float(
            torch.exp(exact_auxiliary_log_ratio).mean()
        ),
        "naive_sum_ratio_mean": float(torch.exp(naive_sum_log_ratio).mean()),
        "official_mean_ratio_mean": float(
            torch.exp(official_mean_log_ratio).mean()
        ),
        "exact_action_ratio_mean_error": abs(
            float(torch.exp(exact_action_log_ratio).mean()) - 1.0
        ),
        "exact_auxiliary_ratio_mean_error": abs(
            float(torch.exp(exact_auxiliary_log_ratio).mean()) - 1.0
        ),
        "naive_sum_ratio_mean_bias": abs(
            float(torch.exp(naive_sum_log_ratio).mean()) - 1.0
        ),
        "official_mean_ratio_mean_bias": abs(
            float(torch.exp(official_mean_log_ratio).mean()) - 1.0
        ),
        "exact_action_score_mean_norm": exact_action_score_norm,
        "exact_auxiliary_score_mean_norm": exact_auxiliary_score_norm,
        "naive_sum_score_mean_norm": naive_sum_score_norm,
        "clip_decision_disagreement": float(
            (exact_clip != official_clip).to(dtype).mean()
        ),
        "exact_clip_rate": float(exact_clip.to(dtype).mean()),
        "official_clip_rate": float(official_clip.to(dtype).mean()),
        "selection_correction_log_ratio_rms": float(
            selection_correction.square().mean().sqrt()
        ),
        "exact_vs_official_policy_gradient_cosine": _cosine_similarity(
            exact_gradient, official_gradient
        ),
        "exact_policy_gradient_norm": float(exact_gradient.norm()),
        "official_policy_gradient_norm": float(official_gradient.norm()),
    }


def run_toy_gate(config: ToyGateConfig) -> dict[str, object]:
    started = time.perf_counter()
    reference = _reference_density_error(config)
    shift_error = _shift_invariance_error(config)
    scenarios = [_run_scenario(config, seed) for seed in config.scenario_seeds]

    def maximum(key: str) -> float:
        return max(float(item[key]) for item in scenarios)

    def minimum(key: str) -> float:
        return min(float(item[key]) for item in scenarios)

    def average(key: str) -> float:
        return sum(float(item[key]) for item in scenarios) / len(scenarios)

    summary = {
        "reference_density_max_abs_error": reference["max_abs_error"],
        "shift_invariance_max_abs_error": shift_error,
        "exact_ratio_mean_error_max": max(
            maximum("exact_action_ratio_mean_error"),
            maximum("exact_auxiliary_ratio_mean_error"),
        ),
        "exact_score_mean_norm_max": max(
            maximum("exact_action_score_mean_norm"),
            maximum("exact_auxiliary_score_mean_norm"),
        ),
        "naive_sum_ratio_mean_bias_min": minimum("naive_sum_ratio_mean_bias"),
        "naive_sum_score_mean_norm_min": minimum("naive_sum_score_mean_norm"),
        "clip_decision_disagreement_mean": average("clip_decision_disagreement"),
        "selection_correction_log_ratio_rms_min": minimum(
            "selection_correction_log_ratio_rms"
        ),
        "exact_vs_official_policy_gradient_cosine_mean": average(
            "exact_vs_official_policy_gradient_cosine"
        ),
        "exact_vs_official_policy_gradient_cosine_min": minimum(
            "exact_vs_official_policy_gradient_cosine"
        ),
    }
    thresholds = config.thresholds
    checks = {
        "boundary_densities_match_known_distributions": summary[
            "reference_density_max_abs_error"
        ]
        <= thresholds["max_reference_density_error"],
        "exact_density_is_shift_invariant": summary[
            "shift_invariance_max_abs_error"
        ]
        <= thresholds["max_shift_invariance_error"],
        "exact_likelihood_ratios_are_normalized": summary[
            "exact_ratio_mean_error_max"
        ]
        <= thresholds["max_exact_ratio_mean_error"],
        "exact_scores_have_zero_mean": summary["exact_score_mean_norm_max"]
        <= thresholds["max_exact_score_mean_norm"],
        "selection_omitting_ratio_is_materially_biased": summary[
            "naive_sum_ratio_mean_bias_min"
        ]
        >= thresholds["min_naive_sum_ratio_mean_bias"],
        "selection_omitting_score_is_materially_biased": summary[
            "naive_sum_score_mean_norm_min"
        ]
        >= thresholds["min_naive_score_mean_norm"],
        "official_and_exact_clipping_decisions_diverge": summary[
            "clip_decision_disagreement_mean"
        ]
        >= thresholds["min_clip_decision_disagreement"],
        "omitted_selection_term_changes_policy_ratios": summary[
            "selection_correction_log_ratio_rms_min"
        ]
        >= thresholds["min_selection_correction_log_ratio_rms"],
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": 1,
        "experiment_name": config.experiment_name,
        "status": status,
        "evidence_level": "synthetic normalization and estimator-validity gate",
        "scientific_limit": (
            "Passing establishes a likelihood defect and a mathematically valid "
            "replacement in the controlled top-k model. It does not establish a "
            "downstream reasoning-quality improvement."
        ),
        "config": {
            "vocabulary_size": config.vocabulary_size,
            "top_k": config.top_k,
            "sample_count": config.sample_count,
            "score_sample_count": config.score_sample_count,
            "scenario_seeds": list(config.scenario_seeds),
            "temperature": config.temperature,
            "gumbel_scale": config.gumbel_scale,
            "logit_scale": config.logit_scale,
            "policy_drift_scale": config.policy_drift_scale,
            "reward_nonlinearity": config.reward_nonlinearity,
            "clip_epsilon": config.clip_epsilon,
        },
        "thresholds": thresholds,
        "checks": checks,
        "summary": summary,
        "reference_diagnostics": reference,
        "scenarios": scenarios,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "wall_time_seconds": time.perf_counter() - started,
        "decision": (
            "advance-to-official-code-replay-audit"
            if status == "pass"
            else "stop-and-reassess-topk-concrete-thesis"
        ),
    }


def _load_config(path: Path) -> ToyGateConfig:
    return ToyGateConfig.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    default_config = Path(__file__).with_name("configs") / "topk_concrete_toy_v1.json"
    parser = argparse.ArgumentParser(description="Run the Top-K Concrete CPU gate")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_toy_gate(_load_config(arguments.config))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(arguments.output)
    print(payload)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
