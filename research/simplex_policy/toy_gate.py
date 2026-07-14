from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .densities import (
    auxiliary_gumbel_log_density,
    auxiliary_gumbel_score,
    concrete_log_density,
    concrete_score,
    effective_sample_size,
    latent_grpo_surrogate_log_prob,
    ppo_clip_mask,
    sample_concrete,
)


@dataclass(frozen=True)
class ToyGateConfig:
    experiment_name: str
    category_count: int
    sample_count: int
    scenario_seeds: tuple[int, ...]
    temperature: float
    logit_scale: float
    policy_drift_scale: float
    reward_nonlinearity: float
    clip_epsilon: float
    gauge_offsets: tuple[float, ...]
    thresholds: dict[str, float]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ToyGateConfig":
        return cls(
            experiment_name=str(raw["experiment_name"]),
            category_count=int(raw["category_count"]),
            sample_count=int(raw["sample_count"]),
            scenario_seeds=tuple(int(value) for value in raw["scenario_seeds"]),
            temperature=float(raw["temperature"]),
            logit_scale=float(raw["logit_scale"]),
            policy_drift_scale=float(raw["policy_drift_scale"]),
            reward_nonlinearity=float(raw["reward_nonlinearity"]),
            clip_epsilon=float(raw["clip_epsilon"]),
            gauge_offsets=tuple(float(value) for value in raw["gauge_offsets"]),
            thresholds={key: float(value) for key, value in raw["thresholds"].items()},
        )


def _trace_variance(samples: Tensor) -> float:
    centered = samples - samples.mean(dim=0, keepdim=True)
    return float(centered.square().sum(dim=-1).mean())


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _relative_error(estimate: Tensor, reference: Tensor) -> float:
    denominator = max(float(reference.norm()), torch.finfo(reference.dtype).eps)
    return float((estimate - reference).norm()) / denominator


def _run_scenario(config: ToyGateConfig, seed: int) -> dict[str, float]:
    dtype = torch.float64
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    old_logits = torch.randn(
        config.category_count, generator=generator, dtype=dtype
    ) * config.logit_scale
    old_logits = old_logits - old_logits.mean()
    drift = torch.randn(
        config.category_count, generator=generator, dtype=dtype
    ) * config.policy_drift_scale
    drift = drift - drift.mean()
    current_logits = old_logits + drift
    utility = torch.randn(
        config.category_count, generator=generator, dtype=dtype
    )
    curvature = torch.randn(
        config.category_count, generator=generator, dtype=dtype
    )

    actions, perturbed_scores = sample_concrete(
        old_logits,
        temperature=config.temperature,
        sample_shape=(config.sample_count,),
        generator=generator,
    )
    rewards = actions @ utility + config.reward_nonlinearity * (
        actions @ curvature
    ).square()
    advantages = rewards - rewards.mean()

    auxiliary_scores = auxiliary_gumbel_score(
        perturbed_scores, old_logits.expand_as(perturbed_scores)
    )
    simplex_scores = concrete_score(
        actions,
        old_logits.expand_as(actions),
        temperature=config.temperature,
    )
    auxiliary_gradient_samples = advantages.unsqueeze(-1) * auxiliary_scores
    simplex_gradient_samples = advantages.unsqueeze(-1) * simplex_scores

    differentiable_logits = old_logits.detach().clone().requires_grad_(True)
    fixed_gumbels = (perturbed_scores - old_logits).detach()
    differentiable_actions = torch.softmax(
        (differentiable_logits + fixed_gumbels) / config.temperature, dim=-1
    )
    differentiable_rewards = differentiable_actions @ utility + config.reward_nonlinearity * (
        differentiable_actions @ curvature
    ).square()
    differentiable_rewards.mean().backward()
    pathwise_gradient = differentiable_logits.grad.detach()

    auxiliary_log_ratio = auxiliary_gumbel_log_density(
        perturbed_scores, current_logits.expand_as(perturbed_scores)
    ) - auxiliary_gumbel_log_density(
        perturbed_scores, old_logits.expand_as(perturbed_scores)
    )
    simplex_log_ratio = concrete_log_density(
        actions,
        current_logits.expand_as(actions),
        temperature=config.temperature,
    ) - concrete_log_density(
        actions,
        old_logits.expand_as(actions),
        temperature=config.temperature,
    )
    auxiliary_clip = ppo_clip_mask(
        auxiliary_log_ratio,
        advantages,
        clip_epsilon=config.clip_epsilon,
    )
    simplex_clip = ppo_clip_mask(
        simplex_log_ratio,
        advantages,
        clip_epsilon=config.clip_epsilon,
    )

    shifted_auxiliary_ratios = []
    for offset in config.gauge_offsets:
        shifted = perturbed_scores + offset
        shifted_auxiliary_ratios.append(
            auxiliary_gumbel_log_density(
                shifted, current_logits.expand_as(shifted)
            )
            - auxiliary_gumbel_log_density(
                shifted, old_logits.expand_as(shifted)
            )
        )
    nuisance_sensitivity = torch.stack(shifted_auxiliary_ratios).std(
        dim=0, unbiased=False
    ).mean()

    return {
        "auxiliary_gradient_relative_error": _relative_error(
            auxiliary_gradient_samples.mean(dim=0), pathwise_gradient
        ),
        "simplex_gradient_relative_error": _relative_error(
            simplex_gradient_samples.mean(dim=0), pathwise_gradient
        ),
        "auxiliary_gradient_trace_variance": _trace_variance(
            auxiliary_gradient_samples
        ),
        "simplex_gradient_trace_variance": _trace_variance(
            simplex_gradient_samples
        ),
        "gradient_variance_reduction": _trace_variance(
            auxiliary_gradient_samples
        )
        / _trace_variance(simplex_gradient_samples),
        "auxiliary_log_ratio_variance": float(auxiliary_log_ratio.var()),
        "simplex_log_ratio_variance": float(simplex_log_ratio.var()),
        "log_ratio_variance_reduction": float(auxiliary_log_ratio.var())
        / float(simplex_log_ratio.var()),
        "auxiliary_effective_sample_size": effective_sample_size(
            auxiliary_log_ratio
        ),
        "simplex_effective_sample_size": effective_sample_size(simplex_log_ratio),
        "effective_sample_size_ratio": effective_sample_size(simplex_log_ratio)
        / effective_sample_size(auxiliary_log_ratio),
        "auxiliary_ratio_mean": float(torch.exp(auxiliary_log_ratio).mean()),
        "simplex_ratio_mean": float(torch.exp(simplex_log_ratio).mean()),
        "auxiliary_clip_rate": float(auxiliary_clip.to(dtype).mean()),
        "simplex_clip_rate": float(simplex_clip.to(dtype).mean()),
        "clip_decision_disagreement": float((auxiliary_clip != simplex_clip).to(dtype).mean()),
        "auxiliary_gauge_log_ratio_sensitivity": float(nuisance_sensitivity),
    }


def _reference_density_error(config: ToyGateConfig) -> float:
    dtype = torch.float64
    logits = torch.linspace(-0.7, 0.8, config.category_count, dtype=dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(8101)
    actions, _ = sample_concrete(
        logits,
        temperature=config.temperature,
        sample_shape=(256,),
        generator=generator,
    )
    ours = concrete_log_density(
        actions,
        logits.expand_as(actions),
        temperature=config.temperature,
    )
    reference_distribution = torch.distributions.RelaxedOneHotCategorical(
        temperature=torch.tensor(config.temperature, dtype=dtype), logits=logits
    )
    reference = reference_distribution.log_prob(actions)
    return float((ours - reference).abs().max())


def _surrogate_diagnostic() -> dict[str, float | bool]:
    current = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    rollout = torch.tensor([-0.5, -0.5], dtype=torch.float64)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
    surrogate = latent_grpo_surrogate_log_prob(current, rollout, advantages)
    component_gradients = []
    for index in range(2):
        gradient = torch.autograd.grad(
            surrogate[index], current, retain_graph=index == 0
        )[0][index]
        component_gradients.append(float(gradient))
    objective_gradients = [
        float(advantages[index]) * component_gradients[index] for index in range(2)
    ]
    return {
        "positive_advantage_logprob_score_gradient": component_gradients[0],
        "negative_advantage_logprob_score_gradient": component_gradients[1],
        "positive_advantage_objective_gradient": objective_gradients[0],
        "negative_advantage_objective_gradient": objective_gradients[1],
        "positive_crossed_margin_misaligned": objective_gradients[0] < 0,
        "negative_crossed_margin_misaligned": objective_gradients[1] > 0,
    }


def run_toy_gate(config: ToyGateConfig) -> dict[str, object]:
    started = time.perf_counter()
    density_error = _reference_density_error(config)
    scenarios = [_run_scenario(config, seed) for seed in config.scenario_seeds]
    variance_reductions = [item["gradient_variance_reduction"] for item in scenarios]
    ratio_variance_reductions = [
        item["log_ratio_variance_reduction"] for item in scenarios
    ]
    ess_ratios = [item["effective_sample_size_ratio"] for item in scenarios]
    clipping_disagreements = [
        item["clip_decision_disagreement"] for item in scenarios
    ]
    simplex_gradient_errors = [
        item["simplex_gradient_relative_error"] for item in scenarios
    ]
    nuisance_sensitivities = [
        item["auxiliary_gauge_log_ratio_sensitivity"] for item in scenarios
    ]
    surrogate = _surrogate_diagnostic()
    summary = {
        "concrete_reference_max_abs_error": density_error,
        "gradient_variance_reduction_geometric_mean": _geometric_mean(
            variance_reductions
        ),
        "gradient_variance_reduction_min": min(variance_reductions),
        "log_ratio_variance_reduction_geometric_mean": _geometric_mean(
            ratio_variance_reductions
        ),
        "effective_sample_size_ratio_geometric_mean": _geometric_mean(ess_ratios),
        "effective_sample_size_ratio_min": min(ess_ratios),
        "clip_decision_disagreement_mean": sum(clipping_disagreements)
        / len(clipping_disagreements),
        "simplex_gradient_relative_error_max": max(simplex_gradient_errors),
        "auxiliary_gauge_log_ratio_sensitivity_mean": sum(nuisance_sensitivities)
        / len(nuisance_sensitivities),
    }
    thresholds = config.thresholds
    checks = {
        "concrete_density_matches_reference": density_error
        <= thresholds["max_concrete_reference_error"],
        "simplex_score_matches_pathwise_gradient": summary[
            "simplex_gradient_relative_error_max"
        ]
        <= thresholds["max_simplex_gradient_relative_error"],
        "material_gradient_variance_reduction": summary[
            "gradient_variance_reduction_geometric_mean"
        ]
        >= thresholds["min_gradient_variance_reduction"],
        "material_ratio_variance_reduction": summary[
            "log_ratio_variance_reduction_geometric_mean"
        ]
        >= thresholds["min_log_ratio_variance_reduction"],
        "material_effective_sample_size_gain": summary[
            "effective_sample_size_ratio_geometric_mean"
        ]
        >= thresholds["min_effective_sample_size_ratio"],
        "operational_clipping_disagreement": summary[
            "clip_decision_disagreement_mean"
        ]
        >= thresholds["min_clip_decision_disagreement"],
        "auxiliary_ratio_has_nuisance_sensitivity": summary[
            "auxiliary_gauge_log_ratio_sensitivity_mean"
        ]
        >= thresholds["min_auxiliary_gauge_sensitivity"],
        "official_surrogate_crossed_margin_diagnostic": bool(
            surrogate["positive_crossed_margin_misaligned"]
        )
        and bool(surrogate["negative_crossed_margin_misaligned"]),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": 1,
        "experiment_name": config.experiment_name,
        "status": status,
        "evidence_level": "synthetic statistical and implementation gate",
        "scientific_limit": (
            "Passing establishes a plausible estimator-level advantage only; it does not "
            "establish improved latent-reasoning training."
        ),
        "config": {
            "category_count": config.category_count,
            "sample_count": config.sample_count,
            "scenario_seeds": list(config.scenario_seeds),
            "temperature": config.temperature,
            "logit_scale": config.logit_scale,
            "policy_drift_scale": config.policy_drift_scale,
            "reward_nonlinearity": config.reward_nonlinearity,
            "clip_epsilon": config.clip_epsilon,
            "gauge_offsets": list(config.gauge_offsets),
        },
        "thresholds": thresholds,
        "checks": checks,
        "summary": summary,
        "surrogate_diagnostic": surrogate,
        "scenarios": scenarios,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "wall_time_seconds": time.perf_counter() - started,
        "decision": (
            "advance-to-real-checkpoint-ratio-audit"
            if status == "pass"
            else "stop-or-redesign-simplex-policy"
        ),
    }


def _load_config(path: Path) -> ToyGateConfig:
    return ToyGateConfig.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    default_config = Path(__file__).with_name("configs") / "simplex_policy_toy_v1.json"
    parser = argparse.ArgumentParser(description="Run the Simplex-GRPO CPU gate")
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
