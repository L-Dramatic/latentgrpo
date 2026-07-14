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

from research.simplex_policy.densities import (
    concrete_log_density,
    concrete_score,
    sample_concrete,
)

from .contracts import ppo_clip_mask, score_mean_diagnostics
from .lepo import lepo_soft_target_score


@dataclass(frozen=True)
class AnalyticAuditConfig:
    experiment_name: str
    sample_count: int
    scenario_seeds: tuple[int, ...]
    vocabulary_size: int
    top_k: int
    temperatures: tuple[float, ...]
    logit_scales: tuple[float, ...]
    policy_drift_scale: float
    clip_epsilon: float
    thresholds: dict[str, float]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "AnalyticAuditConfig":
        config = cls(
            experiment_name=str(raw["experiment_name"]),
            sample_count=int(raw["sample_count"]),
            scenario_seeds=tuple(int(value) for value in raw["scenario_seeds"]),
            vocabulary_size=int(raw["vocabulary_size"]),
            top_k=int(raw["top_k"]),
            temperatures=tuple(float(value) for value in raw["temperatures"]),
            logit_scales=tuple(float(value) for value in raw["logit_scales"]),
            policy_drift_scale=float(raw["policy_drift_scale"]),
            clip_epsilon=float(raw["clip_epsilon"]),
            thresholds={key: float(value) for key, value in raw["thresholds"].items()},
        )
        if config.sample_count < 1000:
            raise ValueError("sample_count must be at least 1000")
        if not config.scenario_seeds or not config.temperatures or not config.logit_scales:
            raise ValueError("scenario axes must be nonempty")
        if config.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least two")
        if not 1 <= config.top_k < config.vocabulary_size:
            raise ValueError("top_k must be smaller than the vocabulary")
        if any(value <= 0 or not math.isfinite(value) for value in config.temperatures):
            raise ValueError("temperatures must be finite and positive")
        if any(value <= 0 or not math.isfinite(value) for value in config.logit_scales):
            raise ValueError("logit scales must be finite and positive")
        if config.policy_drift_scale <= 0 or not 0 < config.clip_epsilon < 1:
            raise ValueError("drift scale and clip epsilon are invalid")
        return config


def _importance_ratio_diagnostics(old_log_density: Tensor, new_log_density: Tensor) -> dict[str, float]:
    ratios = torch.exp(new_log_density - old_log_density)
    mean = ratios.mean()
    standard_error = ratios.std(unbiased=True) / math.sqrt(ratios.numel())
    z_score = float(torch.abs(mean - 1.0) / standard_error.clamp_min(torch.finfo(ratios.dtype).tiny))
    return {
        "mean": float(mean),
        "standard_error": float(standard_error),
        "z_from_one": z_score,
    }


def _run_scenario(
    config: AnalyticAuditConfig,
    *,
    seed: int,
    temperature: float,
    logit_scale: float,
) -> dict[str, Any]:
    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(config.vocabulary_size, generator=generator, dtype=dtype)
    logits = logit_scale * (logits - logits.mean())
    drift = torch.randn(config.vocabulary_size, generator=generator, dtype=dtype)
    drift = config.policy_drift_scale * (drift - drift.mean())
    current_logits = logits + drift
    utility = torch.randn(config.vocabulary_size, generator=generator, dtype=dtype)

    samples, _ = sample_concrete(
        logits,
        temperature=temperature,
        sample_shape=(config.sample_count,),
        generator=generator,
    )
    expanded_old = logits.expand_as(samples)
    expanded_current = current_logits.expand_as(samples)
    topk_probabilities, topk_indices = torch.topk(samples, k=config.top_k, dim=-1)

    exact_scores = concrete_score(samples, expanded_old, temperature=temperature)
    surrogate_scores = lepo_soft_target_score(
        topk_probabilities, topk_indices, expanded_old
    )
    exact_score_stats = score_mean_diagnostics(exact_scores)

    old_log_density = concrete_log_density(samples, expanded_old, temperature=temperature)
    new_log_density = concrete_log_density(samples, expanded_current, temperature=temperature)
    exact_log_ratio = new_log_density - old_log_density

    old_log_probabilities = torch.log_softmax(expanded_old, dim=-1)
    current_log_probabilities = torch.log_softmax(expanded_current, dim=-1)
    selected_old = torch.gather(old_log_probabilities, -1, topk_indices)
    selected_current = torch.gather(current_log_probabilities, -1, topk_indices)
    surrogate_log_ratio = (
        topk_probabilities * (selected_current - selected_old)
    ).sum(dim=-1)

    advantages = samples @ utility
    advantages = advantages - advantages.mean()
    exact_clipping = ppo_clip_mask(
        exact_log_ratio, advantages, epsilon=config.clip_epsilon
    )
    surrogate_clipping = ppo_clip_mask(
        surrogate_log_ratio, advantages, epsilon=config.clip_epsilon
    )

    archive_mass = topk_probabilities.sum(dim=-1)
    score_gap = (surrogate_scores - exact_scores).norm(dim=-1)
    clean_mode = int(torch.argmax(logits))
    proxy_disagreement = (samples.argmax(dim=-1) != clean_mode).to(dtype).mean()

    return {
        "seed": seed,
        "temperature": temperature,
        "logit_scale": logit_scale,
        "archive_tail_mass_mean": float((1.0 - archive_mass).mean()),
        "archive_tail_mass_p95": float(torch.quantile(1.0 - archive_mass, 0.95)),
        "proxy_mode_disagreement": float(proxy_disagreement),
        "exact_score_mean_l2": exact_score_stats.mean_l2,
        "exact_score_snr": exact_score_stats.signal_to_noise,
        "surrogate_score_mean_l2": float(surrogate_scores.mean(dim=0).norm()),
        "surrogate_exact_score_gap_mean": float(score_gap.mean()),
        "exact_ratio": _importance_ratio_diagnostics(old_log_density, new_log_density),
        "surrogate_exp_ratio_mean": float(torch.exp(surrogate_log_ratio).mean()),
        "clip_decision_disagreement": float((exact_clipping != surrogate_clipping).to(dtype).mean()),
        "exact_clip_fraction": float(exact_clipping.to(dtype).mean()),
        "surrogate_clip_fraction": float(surrogate_clipping.to(dtype).mean()),
    }


def run_audit(config: AnalyticAuditConfig) -> dict[str, Any]:
    started = time.time()
    scenarios = [
        _run_scenario(
            config,
            seed=seed,
            temperature=temperature,
            logit_scale=logit_scale,
        )
        for seed in config.scenario_seeds
        for temperature in config.temperatures
        for logit_scale in config.logit_scales
    ]
    thresholds = config.thresholds
    aggregates = {
        "scenario_count": len(scenarios),
        "exact_score_snr_max": max(row["exact_score_snr"] for row in scenarios),
        "exact_ratio_z_max": max(row["exact_ratio"]["z_from_one"] for row in scenarios),
        "archive_tail_mass_mean_min": min(row["archive_tail_mass_mean"] for row in scenarios),
        "proxy_mode_disagreement_min": min(row["proxy_mode_disagreement"] for row in scenarios),
        "surrogate_exact_score_gap_mean_min": min(
            row["surrogate_exact_score_gap_mean"] for row in scenarios
        ),
        "clip_decision_disagreement_mean": sum(
            row["clip_decision_disagreement"] for row in scenarios
        )
        / len(scenarios),
    }
    checks = {
        "exact_score_zero_mean_control": aggregates["exact_score_snr_max"]
        <= thresholds["exact_score_snr_max"],
        "exact_ratio_normalization_control": aggregates["exact_ratio_z_max"]
        <= thresholds["exact_ratio_z_max"],
        "archive_tail_loss_detected": aggregates["archive_tail_mass_mean_min"]
        >= thresholds["archive_tail_mass_mean_min"],
        "proxy_execution_divergence_detected": aggregates["proxy_mode_disagreement_min"]
        >= thresholds["proxy_mode_disagreement_min"],
        "surrogate_is_not_exact_score_detected": aggregates[
            "surrogate_exact_score_gap_mean_min"
        ]
        >= thresholds["surrogate_exact_score_gap_mean_min"],
        "optimizer_decision_difference_detected": aggregates[
            "clip_decision_disagreement_mean"
        ]
        >= thresholds["clip_decision_disagreement_mean_min"],
    }
    return {
        "experiment_name": config.experiment_name,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "aggregates": aggregates,
        "scenarios": scenarios,
        "runtime": {
            "seconds": time.time() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }


def _canonical_config_hash(raw: dict[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    config = AnalyticAuditConfig.from_mapping(raw)
    report = run_audit(config)
    report["config"] = raw
    report["config_sha256"] = _canonical_config_hash(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "checks": report["checks"],
        "aggregates": report["aggregates"],
        "config_sha256": report["config_sha256"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
