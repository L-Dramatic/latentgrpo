from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .policies import (
    score_squashed_selected_log_density,
    score_squashed_topk_from_gumbels,
)


@dataclass(frozen=True)
class SupportConcentrationGateConfig:
    experiment_name: str
    vocabulary_size: int
    top_k: int
    sample_count: int
    score_sample_count: int
    scenario_seeds: tuple[int, ...]
    logit_scales: tuple[float, ...]
    temperature: float
    gumbel_scale: float
    lower_bound: float
    upper_bound: float
    policy_drift_scale: float
    saturation_derivative_threshold: float
    thresholds: dict[str, float]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SupportConcentrationGateConfig":
        config = cls(
            experiment_name=str(raw["experiment_name"]),
            vocabulary_size=int(raw["vocabulary_size"]),
            top_k=int(raw["top_k"]),
            sample_count=int(raw["sample_count"]),
            score_sample_count=int(raw["score_sample_count"]),
            scenario_seeds=tuple(int(value) for value in raw["scenario_seeds"]),
            logit_scales=tuple(float(value) for value in raw["logit_scales"]),
            temperature=float(raw["temperature"]),
            gumbel_scale=float(raw["gumbel_scale"]),
            lower_bound=float(raw["lower_bound"]),
            upper_bound=float(raw["upper_bound"]),
            policy_drift_scale=float(raw["policy_drift_scale"]),
            saturation_derivative_threshold=float(
                raw["saturation_derivative_threshold"]
            ),
            thresholds={key: float(value) for key, value in raw["thresholds"].items()},
        )
        if config.vocabulary_size < 2 or not 1 <= config.top_k <= config.vocabulary_size:
            raise ValueError("vocabulary size and top_k are inconsistent")
        if config.sample_count < 2 or not 1 <= config.score_sample_count <= config.sample_count:
            raise ValueError("sample counts are inconsistent")
        if not config.scenario_seeds or not config.logit_scales:
            raise ValueError("seeds and logit scales must be nonempty")
        if config.temperature <= 0 or config.gumbel_scale <= 0:
            raise ValueError("temperatures and scales must be positive")
        if config.lower_bound >= config.upper_bound:
            raise ValueError("lower_bound must be below upper_bound")
        if config.policy_drift_scale <= 0:
            raise ValueError("policy_drift_scale must be positive")
        if not 0 < config.saturation_derivative_threshold < 1:
            raise ValueError("saturation derivative threshold must be in (0,1)")
        return config


@dataclass(frozen=True)
class BaselineSample:
    ordered_indices: Tensor
    selected_scores: Tensor
    weights: Tensor
    selected_raw_gumbels: Tensor


def _standard_gumbels(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tensor:
    uniforms = torch.rand(shape, dtype=dtype, generator=generator)
    epsilon = torch.finfo(dtype).eps
    uniforms = uniforms.clamp(min=epsilon, max=1.0 - epsilon)
    return -torch.log(-torch.log(uniforms))


def _baseline_from_gumbels(
    canonical_logits: Tensor,
    raw_gumbels: Tensor,
    *,
    top_k: int,
    temperature: float,
    gumbel_scale: float,
    clip_bounds: tuple[float, float] | None,
) -> BaselineSample:
    effective_gumbels = raw_gumbels
    if clip_bounds is not None:
        effective_gumbels = raw_gumbels.clamp(*clip_bounds)
    scores = canonical_logits + gumbel_scale * effective_gumbels
    selected_scores, ordered_indices = torch.topk(
        scores, k=top_k, dim=-1, sorted=True
    )
    weights = torch.softmax(selected_scores / temperature, dim=-1)
    selected_raw_gumbels = torch.gather(raw_gumbels, -1, ordered_indices)
    return BaselineSample(
        ordered_indices=ordered_indices,
        selected_scores=selected_scores,
        weights=weights,
        selected_raw_gumbels=selected_raw_gumbels,
    )


def _mixture_entropy(weights: Tensor) -> Tensor:
    return -(weights * torch.log(weights)).sum(dim=-1)


def _support_inclusion_entropy(indices: Tensor, vocabulary_size: int) -> float:
    counts = torch.bincount(indices.reshape(-1), minlength=vocabulary_size).to(
        torch.float64
    )
    probabilities = counts / counts.sum()
    positive = probabilities > 0
    return float(-(probabilities[positive] * torch.log(probabilities[positive])).sum())


def _score_mean_norm(
    logits: Tensor,
    squashed_scores: Tensor,
    ordered_indices: Tensor,
    config: SupportConcentrationGateConfig,
) -> float:
    differentiable_logits = logits.detach().clone().requires_grad_(True)
    mean_density = score_squashed_selected_log_density(
        squashed_scores,
        ordered_indices,
        differentiable_logits,
        gumbel_scale=config.gumbel_scale,
        lower=config.lower_bound,
        upper=config.upper_bound,
    ).mean()
    gradient = torch.autograd.grad(mean_density, differentiable_logits)[0]
    return float(gradient.norm())


def _run_scenario(
    config: SupportConcentrationGateConfig,
    *,
    seed: int,
    scenario_seed: int,
    logit_scale: float,
) -> dict[str, float | int]:
    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(scenario_seed)
    logits = torch.randn(
        config.vocabulary_size, dtype=dtype, generator=generator
    ) * logit_scale
    logits = logits - logits.mean()
    canonical = torch.log_softmax(logits, dim=-1)
    raw_gumbels = _standard_gumbels(
        (config.sample_count, config.vocabulary_size),
        dtype=dtype,
        generator=generator,
    )

    unbounded = _baseline_from_gumbels(
        canonical,
        raw_gumbels,
        top_k=config.top_k,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
        clip_bounds=None,
    )
    hard_clipped = _baseline_from_gumbels(
        canonical,
        raw_gumbels,
        top_k=config.top_k,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
        clip_bounds=(config.lower_bound, config.upper_bound),
    )
    squashed = score_squashed_topk_from_gumbels(
        logits,
        raw_gumbels,
        top_k=config.top_k,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
        lower=config.lower_bound,
        upper=config.upper_bound,
    )

    squash_support_mismatch = (
        squashed.ordered_indices != unbounded.ordered_indices
    ).any(dim=-1)
    hard_support_change = (
        hard_clipped.ordered_indices != unbounded.ordered_indices
    ).any(dim=-1)
    support_matches = (
        hard_clipped.ordered_indices.unsqueeze(-1)
        == unbounded.ordered_indices.unsqueeze(-2)
    )
    hard_support_overlap = support_matches.any(dim=-1).to(dtype).mean(dim=-1)

    drift = torch.randn(
        config.vocabulary_size, dtype=dtype, generator=generator
    )
    drift = drift - drift.mean()
    drift = drift / drift.square().mean().sqrt()
    current_logits = logits + config.policy_drift_scale * drift
    old_density = score_squashed_selected_log_density(
        squashed.squashed_scores,
        squashed.ordered_indices,
        logits,
        gumbel_scale=config.gumbel_scale,
        lower=config.lower_bound,
        upper=config.upper_bound,
    )
    current_density = score_squashed_selected_log_density(
        squashed.squashed_scores,
        squashed.ordered_indices,
        current_logits,
        gumbel_scale=config.gumbel_scale,
        lower=config.lower_bound,
        upper=config.upper_bound,
    )
    ratio_mean = float(torch.exp(current_density - old_density).mean())

    half_range = 0.5 * (config.upper_bound - config.lower_bound)
    squash_derivative = 1.0 - torch.tanh(
        squashed.raw_selected_scores / half_range
    ).square()
    hard_entropy = _mixture_entropy(hard_clipped.weights)
    squashed_entropy = _mixture_entropy(squashed.weights)
    hard_qmax = hard_clipped.weights.max(dim=-1).values
    squashed_qmax = squashed.weights.max(dim=-1).values
    hard_support_entropy = _support_inclusion_entropy(
        hard_clipped.ordered_indices, config.vocabulary_size
    )
    squashed_support_entropy = _support_inclusion_entropy(
        squashed.ordered_indices, config.vocabulary_size
    )

    theoretical_qmax = 1.0 / (
        1.0
        + (config.top_k - 1)
        * math.exp(
            -(config.upper_bound - config.lower_bound) / config.temperature
        )
    )
    score_count = config.score_sample_count
    score_mean_norm = _score_mean_norm(
        logits,
        squashed.squashed_scores[:score_count],
        squashed.ordered_indices[:score_count],
        config,
    )
    return {
        "seed": seed,
        "scenario_seed": scenario_seed,
        "logit_scale": logit_scale,
        "squashed_unbounded_support_mismatch_rate": float(
            squash_support_mismatch.to(dtype).mean()
        ),
        "hardclip_unbounded_support_change_rate": float(
            hard_support_change.to(dtype).mean()
        ),
        "hardclip_unbounded_support_overlap": float(hard_support_overlap.mean()),
        "hardclip_selected_upper_atom_fraction": float(
            (
                hard_clipped.selected_raw_gumbels >= config.upper_bound
            ).to(dtype).mean()
        ),
        "selected_squash_saturation_fraction": float(
            (
                squash_derivative < config.saturation_derivative_threshold
            ).to(dtype).mean()
        ),
        "selected_squash_derivative_mean": float(squash_derivative.mean()),
        "hardclip_mixture_entropy_mean": float(hard_entropy.mean()),
        "squashed_mixture_entropy_mean": float(squashed_entropy.mean()),
        "hardclip_squashed_mixture_entropy_gap": abs(
            float(hard_entropy.mean() - squashed_entropy.mean())
        ),
        "hardclip_qmax_p95": float(torch.quantile(hard_qmax, 0.95)),
        "squashed_qmax_p95": float(torch.quantile(squashed_qmax, 0.95)),
        "hardclip_squashed_qmax_p95_gap": abs(
            float(torch.quantile(hard_qmax, 0.95) - torch.quantile(squashed_qmax, 0.95))
        ),
        "hardclip_support_inclusion_entropy": hard_support_entropy,
        "squashed_support_inclusion_entropy": squashed_support_entropy,
        "support_inclusion_entropy_gain": (
            squashed_support_entropy - hard_support_entropy
        ),
        "squashed_qmax_observed": float(squashed_qmax.max()),
        "squashed_qmax_theoretical_bound": theoretical_qmax,
        "theoretical_qmax_bound_violation": max(
            0.0, float(squashed_qmax.max()) - theoretical_qmax
        ),
        "exact_ratio_mean": ratio_mean,
        "exact_ratio_mean_error": abs(ratio_mean - 1.0),
        "exact_score_mean_norm": score_mean_norm,
    }


def run_support_concentration_gate(
    config: SupportConcentrationGateConfig,
) -> dict[str, object]:
    started = time.perf_counter()
    scenarios: list[dict[str, float | int]] = []
    scenario_index = 0
    for logit_scale in config.logit_scales:
        for seed in config.scenario_seeds:
            scenarios.append(
                _run_scenario(
                    config,
                    seed=seed,
                    scenario_seed=seed + 100_000 * scenario_index,
                    logit_scale=logit_scale,
                )
            )
            scenario_index += 1

    def values(key: str) -> Tensor:
        return torch.tensor(
            [float(item[key]) for item in scenarios], dtype=torch.float64
        )

    entropy_gains = values("support_inclusion_entropy_gain")
    summary = {
        "scenario_count": len(scenarios),
        "squashed_unbounded_support_mismatch_rate_max": float(
            values("squashed_unbounded_support_mismatch_rate").max()
        ),
        "exact_ratio_mean_error_max": float(values("exact_ratio_mean_error").max()),
        "exact_score_mean_norm_max": float(values("exact_score_mean_norm").max()),
        "hardclip_unbounded_support_change_rate_min": float(
            values("hardclip_unbounded_support_change_rate").min()
        ),
        "hardclip_unbounded_support_change_rate_mean": float(
            values("hardclip_unbounded_support_change_rate").mean()
        ),
        "hardclip_selected_upper_atom_fraction_min": float(
            values("hardclip_selected_upper_atom_fraction").min()
        ),
        "selected_squash_saturation_fraction_max": float(
            values("selected_squash_saturation_fraction").max()
        ),
        "hardclip_squashed_mixture_entropy_gap_max": float(
            values("hardclip_squashed_mixture_entropy_gap").max()
        ),
        "hardclip_squashed_qmax_p95_gap_max": float(
            values("hardclip_squashed_qmax_p95_gap").max()
        ),
        "support_inclusion_entropy_gain_mean": float(entropy_gains.mean()),
        "support_inclusion_entropy_gain_positive_fraction": float(
            (entropy_gains > 0).to(torch.float64).mean()
        ),
        "theoretical_qmax_bound_violation_max": float(
            values("theoretical_qmax_bound_violation").max()
        ),
    }
    thresholds = config.thresholds
    checks = {
        "score_squash_preserves_unbounded_ordered_support": summary[
            "squashed_unbounded_support_mismatch_rate_max"
        ]
        <= thresholds["max_squashed_unbounded_support_mismatch_rate"],
        "exact_likelihood_has_calibrated_ratios_and_scores": summary[
            "exact_ratio_mean_error_max"
        ]
        <= thresholds["max_exact_ratio_mean_error"]
        and summary["exact_score_mean_norm_max"]
        <= thresholds["max_exact_score_mean_norm"],
        "hard_clipping_has_material_atoms_and_support_distortion": summary[
            "hardclip_unbounded_support_change_rate_min"
        ]
        >= thresholds["min_hardclip_unbounded_support_change_rate"]
        and summary["hardclip_selected_upper_atom_fraction_min"]
        >= thresholds["min_hardclip_selected_upper_atom_fraction"],
        "selected_scores_avoid_widespread_squash_saturation": summary[
            "selected_squash_saturation_fraction_max"
        ]
        <= thresholds["max_selected_squash_saturation_fraction"],
        "squashed_policy_matches_hardclip_mixture_concentration": summary[
            "hardclip_squashed_mixture_entropy_gap_max"
        ]
        <= thresholds["max_hardclip_squashed_mixture_entropy_gap"]
        and summary["hardclip_squashed_qmax_p95_gap_max"]
        <= thresholds["max_hardclip_squashed_qmax_p95_gap"],
        "squashed_policy_improves_support_inclusion_diversity": summary[
            "support_inclusion_entropy_gain_mean"
        ]
        >= thresholds["min_support_inclusion_entropy_gain_mean"]
        and summary["support_inclusion_entropy_gain_positive_fraction"]
        >= thresholds["min_support_inclusion_entropy_gain_positive_fraction"],
        "executed_mixture_obeys_theoretical_weight_bound": summary[
            "theoretical_qmax_bound_violation_max"
        ]
        <= thresholds["max_theoretical_qmax_bound_violation"],
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": 1,
        "experiment_name": config.experiment_name,
        "status": status,
        "evidence_level": "synthetic support-preservation and concentration gate",
        "scientific_limit": (
            "Passing shows a sampler-level Pareto candidate under synthetic "
            "logits. It does not establish valid latent states or downstream "
            "reasoning gains on a checkpoint."
        ),
        "config": {
            "vocabulary_size": config.vocabulary_size,
            "top_k": config.top_k,
            "sample_count": config.sample_count,
            "score_sample_count": config.score_sample_count,
            "scenario_seeds": list(config.scenario_seeds),
            "logit_scales": list(config.logit_scales),
            "temperature": config.temperature,
            "gumbel_scale": config.gumbel_scale,
            "lower_bound": config.lower_bound,
            "upper_bound": config.upper_bound,
            "policy_drift_scale": config.policy_drift_scale,
            "saturation_derivative_threshold": config.saturation_derivative_threshold,
        },
        "thresholds": thresholds,
        "checks": checks,
        "summary": summary,
        "scenarios": scenarios,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "wall_time_seconds": time.perf_counter() - started,
        "decision": (
            "advance-to-public-checkpoint-rollout-preregistration"
            if status == "pass"
            else "stop-score-squashed-method-and-preserve-theory"
        ),
    }


def _load_config(path: Path) -> tuple[SupportConcentrationGateConfig, str]:
    payload = path.read_bytes()
    return (
        SupportConcentrationGateConfig.from_mapping(json.loads(payload)),
        hashlib.sha256(payload).hexdigest(),
    )


def main() -> int:
    default_config = (
        Path(__file__).with_name("configs") / "support_concentration_gate_v1.json"
    )
    parser = argparse.ArgumentParser(description="Run the SSG-PO synthetic gate")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    config, config_sha256 = _load_config(arguments.config)
    result = run_support_concentration_gate(config)
    result["config_sha256"] = config_sha256
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
