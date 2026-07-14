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
import torch.nn.functional as F
from torch import Tensor

from research.simplex_policy.densities import (
    concrete_log_density,
    concrete_score,
    sample_concrete,
)

from .contracts import score_mean_diagnostics
from .lepo import apply_lepo_sampling_filters, lepo_soft_target_score


@dataclass(frozen=True)
class SourceFaithfulAuditConfig:
    experiment_name: str
    sample_count: int
    scenario_seeds: tuple[int, ...]
    vocabulary_size: int
    top_k: int
    top_p: float
    temperatures: tuple[float, ...]
    logit_scales: tuple[float, ...]
    policy_drift_scale: float
    thresholds: dict[str, float]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SourceFaithfulAuditConfig":
        config = cls(
            experiment_name=str(raw["experiment_name"]),
            sample_count=int(raw["sample_count"]),
            scenario_seeds=tuple(int(value) for value in raw["scenario_seeds"]),
            vocabulary_size=int(raw["vocabulary_size"]),
            top_k=int(raw["top_k"]),
            top_p=float(raw["top_p"]),
            temperatures=tuple(float(value) for value in raw["temperatures"]),
            logit_scales=tuple(float(value) for value in raw["logit_scales"]),
            policy_drift_scale=float(raw["policy_drift_scale"]),
            thresholds={key: float(value) for key, value in raw["thresholds"].items()},
        )
        if config.sample_count < 1000:
            raise ValueError("sample_count must be at least 1000")
        if not config.scenario_seeds or not config.temperatures or not config.logit_scales:
            raise ValueError("scenario axes must be nonempty")
        if config.vocabulary_size < 2 or not 1 <= config.top_k < config.vocabulary_size:
            raise ValueError("vocabulary_size and top_k are inconsistent")
        if not 0.0 < config.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if any(value <= 0 or not math.isfinite(value) for value in config.temperatures):
            raise ValueError("temperatures must be finite and positive")
        if any(value <= 0 or not math.isfinite(value) for value in config.logit_scales):
            raise ValueError("logit scales must be finite and positive")
        if config.policy_drift_scale <= 0 or not math.isfinite(config.policy_drift_scale):
            raise ValueError("policy_drift_scale must be finite and positive")
        return config


def _importance_ratio_z(old_log_density: Tensor, new_log_density: Tensor) -> tuple[float, float, float]:
    ratios = torch.exp(new_log_density - old_log_density)
    mean = ratios.mean()
    standard_error = ratios.std(unbiased=True) / math.sqrt(ratios.numel())
    z_score = torch.abs(mean - 1.0) / standard_error.clamp_min(
        torch.finfo(ratios.dtype).tiny
    )
    return float(mean), float(standard_error), float(z_score)


def _run_scenario(
    config: SourceFaithfulAuditConfig,
    *,
    seed: int,
    temperature: float,
    logit_scale: float,
) -> dict[str, Any]:
    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw_logits = torch.randn(config.vocabulary_size, generator=generator, dtype=dtype)
    raw_logits = logit_scale * (raw_logits - raw_logits.mean())
    drift = torch.randn(config.vocabulary_size, generator=generator, dtype=dtype)
    drift = config.policy_drift_scale * (drift - drift.mean())
    current_raw_logits = raw_logits + drift

    filtered_logits = apply_lepo_sampling_filters(
        raw_logits, top_k=config.top_k, top_p=config.top_p
    )
    active = torch.isfinite(filtered_logits)
    active_logits = filtered_logits[active]
    active_size = int(active.sum())
    active_samples, _ = sample_concrete(
        active_logits,
        temperature=temperature,
        sample_shape=(config.sample_count,),
        generator=generator,
    )
    samples = torch.zeros(
        config.sample_count, config.vocabulary_size, dtype=dtype
    )
    samples[:, active] = active_samples
    archived_probabilities, archived_indices = torch.topk(
        samples, k=config.top_k, dim=-1
    )
    reconstructed = torch.zeros_like(samples).scatter(
        -1, archived_indices, archived_probabilities
    )

    exact_active_scores = concrete_score(
        active_samples,
        active_logits.expand_as(active_samples),
        temperature=temperature,
    )
    exact_scores = torch.zeros_like(samples)
    exact_scores[:, active] = exact_active_scores
    surrogate_scores = lepo_soft_target_score(
        archived_probabilities,
        archived_indices,
        raw_logits.expand_as(samples),
    )
    exact_diagnostics = score_mean_diagnostics(exact_active_scores)
    surrogate_diagnostics = score_mean_diagnostics(surrogate_scores)

    current_active_logits = current_raw_logits[active]
    old_log_density = concrete_log_density(
        active_samples,
        active_logits.expand_as(active_samples),
        temperature=temperature,
    )
    new_log_density = concrete_log_density(
        active_samples,
        current_active_logits.expand_as(active_samples),
        temperature=temperature,
    )
    ratio_mean, ratio_se, ratio_z = _importance_ratio_z(
        old_log_density, new_log_density
    )

    current_dynamic_filtered = apply_lepo_sampling_filters(
        current_raw_logits, top_k=config.top_k, top_p=config.top_p
    )
    current_active = torch.isfinite(current_dynamic_filtered)
    support_union = active | current_active
    support_symmetric_difference = active ^ current_active
    support_churn = support_symmetric_difference.sum() / support_union.sum()

    score_gap = (surrogate_scores - exact_scores).norm(dim=-1)
    score_cosine = F.cosine_similarity(surrogate_scores, exact_scores, dim=-1)
    archive_mass_error = torch.abs(archived_probabilities.sum(dim=-1) - 1.0)
    proxy_mode = int(torch.argmax(filtered_logits))
    proxy_disagreement = (samples.argmax(dim=-1) != proxy_mode).to(dtype).mean()
    full_model_probabilities = torch.softmax(raw_logits, dim=-1)
    excluded_model_mass = 1.0 - full_model_probabilities[active].sum()

    return {
        "seed": seed,
        "temperature": temperature,
        "logit_scale": logit_scale,
        "active_support_size": active_size,
        "archive_mass_error_max": float(archive_mass_error.max()),
        "reconstruction_l1_max": float((samples - reconstructed).abs().sum(dim=-1).max()),
        "excluded_full_model_mass": float(excluded_model_mass),
        "proxy_mode_disagreement": float(proxy_disagreement),
        "exact_score_snr": exact_diagnostics.signal_to_noise,
        "surrogate_score_snr": surrogate_diagnostics.signal_to_noise,
        "surrogate_score_mean_l2": surrogate_diagnostics.mean_l2,
        "surrogate_exact_score_gap_mean": float(score_gap.mean()),
        "surrogate_exact_score_cosine_mean": float(score_cosine.mean()),
        "exact_ratio_mean": ratio_mean,
        "exact_ratio_standard_error": ratio_se,
        "exact_ratio_z_from_one": ratio_z,
        "dynamic_support_churn": float(support_churn),
    }


def run_audit(config: SourceFaithfulAuditConfig) -> dict[str, Any]:
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
    aggregates = {
        "scenario_count": len(scenarios),
        "archive_mass_error_max": max(row["archive_mass_error_max"] for row in scenarios),
        "reconstruction_l1_max": max(row["reconstruction_l1_max"] for row in scenarios),
        "active_support_size_max": max(row["active_support_size"] for row in scenarios),
        "active_support_size_min": min(row["active_support_size"] for row in scenarios),
        "exact_score_snr_max": max(row["exact_score_snr"] for row in scenarios),
        "exact_ratio_z_max": max(row["exact_ratio_z_from_one"] for row in scenarios),
        "proxy_mode_disagreement_min": min(row["proxy_mode_disagreement"] for row in scenarios),
        "surrogate_score_snr_min": min(row["surrogate_score_snr"] for row in scenarios),
        "surrogate_exact_score_gap_mean_min": min(
            row["surrogate_exact_score_gap_mean"] for row in scenarios
        ),
        "surrogate_exact_score_cosine_mean": sum(
            row["surrogate_exact_score_cosine_mean"] for row in scenarios
        )
        / len(scenarios),
        "dynamic_support_churn_mean": sum(
            row["dynamic_support_churn"] for row in scenarios
        )
        / len(scenarios),
    }
    thresholds = config.thresholds
    checks = {
        "archive_is_normalized": aggregates["archive_mass_error_max"]
        <= thresholds["archive_mass_error_max"],
        "archive_reconstructs_action": aggregates["reconstruction_l1_max"]
        <= thresholds["reconstruction_l1_max"],
        "sampler_support_is_bounded_by_archive": aggregates["active_support_size_max"]
        <= thresholds["active_support_size_max"],
        "exact_score_zero_mean_control": aggregates["exact_score_snr_max"]
        <= thresholds["exact_score_snr_max"],
        "exact_ratio_normalization_control": aggregates["exact_ratio_z_max"]
        <= thresholds["exact_ratio_z_max"],
        "proxy_execution_divergence_detected": aggregates["proxy_mode_disagreement_min"]
        >= thresholds["proxy_mode_disagreement_min"],
        "surrogate_nonzero_mean_detected": aggregates["surrogate_score_snr_min"]
        >= thresholds["surrogate_score_snr_min"],
        "surrogate_is_not_exact_score_detected": aggregates[
            "surrogate_exact_score_gap_mean_min"
        ]
        >= thresholds["surrogate_exact_score_gap_mean_min"],
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


def _canonical_hash(raw: dict[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    config = SourceFaithfulAuditConfig.from_mapping(raw)
    report = run_audit(config)
    report["config"] = raw
    report["config_sha256"] = _canonical_hash(raw)
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
