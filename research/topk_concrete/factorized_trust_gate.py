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

from .densities import (
    conditional_topk_weight_log_density,
    ordered_plackett_luce_log_probability,
    sample_topk_concrete,
    topk_concrete_log_density,
)


@dataclass(frozen=True)
class FactorizedTrustGateConfig:
    experiment_name: str
    vocabulary_size: int
    top_k_values: tuple[int, ...]
    sample_count: int
    calibration_sample_count: int
    scenario_seeds: tuple[int, ...]
    temperature: float
    gumbel_scale: float
    logit_scales: tuple[float, ...]
    drift_modes: tuple[str, ...]
    target_joint_kl: float
    calibration_iterations: int
    maximum_drift_scale: float
    clip_epsilon: float
    thresholds: dict[str, float]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "FactorizedTrustGateConfig":
        config = cls(
            experiment_name=str(raw["experiment_name"]),
            vocabulary_size=int(raw["vocabulary_size"]),
            top_k_values=tuple(int(value) for value in raw["top_k_values"]),
            sample_count=int(raw["sample_count"]),
            calibration_sample_count=int(raw["calibration_sample_count"]),
            scenario_seeds=tuple(int(value) for value in raw["scenario_seeds"]),
            temperature=float(raw["temperature"]),
            gumbel_scale=float(raw["gumbel_scale"]),
            logit_scales=tuple(float(value) for value in raw["logit_scales"]),
            drift_modes=tuple(str(value) for value in raw["drift_modes"]),
            target_joint_kl=float(raw["target_joint_kl"]),
            calibration_iterations=int(raw["calibration_iterations"]),
            maximum_drift_scale=float(raw["maximum_drift_scale"]),
            clip_epsilon=float(raw["clip_epsilon"]),
            thresholds={key: float(value) for key, value in raw["thresholds"].items()},
        )
        if config.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least two")
        if not config.top_k_values or any(
            value < 1 or value > config.vocabulary_size
            for value in config.top_k_values
        ):
            raise ValueError("every top_k value must fit within the vocabulary")
        if config.sample_count < 1 or config.calibration_sample_count < 1:
            raise ValueError("sample counts must be positive")
        if not config.scenario_seeds or not config.logit_scales:
            raise ValueError("seeds and logit scales must be nonempty")
        if set(config.drift_modes) - {"random", "sharpen", "flatten"}:
            raise ValueError("unsupported drift mode")
        if not 0 < config.target_joint_kl or config.calibration_iterations < 1:
            raise ValueError("KL target and calibration iterations must be positive")
        if config.maximum_drift_scale <= 0 or not 0 < config.clip_epsilon < 1:
            raise ValueError("drift scale and clip epsilon are invalid")
        return config


def _center_and_normalize(values: Tensor) -> Tensor:
    centered = values - values.mean()
    root_mean_square = centered.square().mean().sqrt()
    if float(root_mean_square) == 0.0:
        raise ValueError("cannot normalize a constant drift direction")
    return centered / root_mean_square


def _make_drift_direction(
    old_logits: Tensor,
    mode: str,
    *,
    generator: torch.Generator,
) -> Tensor:
    if mode == "random":
        return _center_and_normalize(
            torch.randn(old_logits.shape, dtype=old_logits.dtype, generator=generator)
        )
    standardized_logits = _center_and_normalize(old_logits)
    if mode == "sharpen":
        return standardized_logits
    if mode == "flatten":
        return -standardized_logits
    raise ValueError(f"unsupported drift mode: {mode}")


def _factor_log_densities(
    weights: Tensor,
    ordered_indices: Tensor,
    logits: Tensor,
    config: FactorizedTrustGateConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    joint = topk_concrete_log_density(
        weights,
        ordered_indices,
        logits,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
    )
    support = ordered_plackett_luce_log_probability(
        ordered_indices,
        logits,
        gumbel_scale=config.gumbel_scale,
    )
    conditional = conditional_topk_weight_log_density(
        weights,
        ordered_indices,
        logits,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
    )
    return joint, support, conditional


def _calibrate_drift_scale(
    old_logits: Tensor,
    drift_direction: Tensor,
    top_k: int,
    config: FactorizedTrustGateConfig,
    *,
    generator: torch.Generator,
) -> tuple[float, float, bool]:
    samples = sample_topk_concrete(
        old_logits,
        top_k=top_k,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
        sample_shape=(config.calibration_sample_count,),
        generator=generator,
    )
    old_joint = topk_concrete_log_density(
        samples.weights,
        samples.ordered_indices,
        old_logits,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
    )

    def estimate_joint_kl(scale: float) -> float:
        current_joint = topk_concrete_log_density(
            samples.weights,
            samples.ordered_indices,
            old_logits + scale * drift_direction,
            temperature=config.temperature,
            gumbel_scale=config.gumbel_scale,
        )
        return float((old_joint - current_joint).mean())

    lower = 0.0
    upper = min(0.05, config.maximum_drift_scale)
    upper_kl = estimate_joint_kl(upper)
    while upper_kl < config.target_joint_kl and upper < config.maximum_drift_scale:
        lower = upper
        upper = min(2.0 * upper, config.maximum_drift_scale)
        upper_kl = estimate_joint_kl(upper)

    if upper_kl < config.target_joint_kl:
        return upper, upper_kl, False

    best_scale = upper
    best_kl = upper_kl
    for _ in range(config.calibration_iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_kl = estimate_joint_kl(midpoint)
        if abs(midpoint_kl - config.target_joint_kl) < abs(
            best_kl - config.target_joint_kl
        ):
            best_scale = midpoint
            best_kl = midpoint_kl
        if midpoint_kl < config.target_joint_kl:
            lower = midpoint
        else:
            upper = midpoint
    return best_scale, best_kl, True


def _standard_error(values: Tensor) -> float:
    if values.numel() < 2:
        return 0.0
    return float(values.std(unbiased=True) / math.sqrt(values.numel()))


def _run_scenario(
    config: FactorizedTrustGateConfig,
    *,
    seed: int,
    scenario_seed: int,
    logit_scale: float,
    top_k: int,
    drift_mode: str,
) -> dict[str, float | int | str | bool]:
    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(scenario_seed)
    old_logits = torch.randn(
        config.vocabulary_size, dtype=dtype, generator=generator
    ) * logit_scale
    old_logits = old_logits - old_logits.mean()
    drift_direction = _make_drift_direction(
        old_logits, drift_mode, generator=generator
    )
    drift_scale, calibration_kl, calibrated = _calibrate_drift_scale(
        old_logits,
        drift_direction,
        top_k,
        config,
        generator=generator,
    )
    current_logits = old_logits + drift_scale * drift_direction

    samples = sample_topk_concrete(
        old_logits,
        top_k=top_k,
        temperature=config.temperature,
        gumbel_scale=config.gumbel_scale,
        sample_shape=(config.sample_count,),
        generator=generator,
    )
    old_joint, old_support, old_conditional = _factor_log_densities(
        samples.weights, samples.ordered_indices, old_logits, config
    )
    current_joint, current_support, current_conditional = _factor_log_densities(
        samples.weights, samples.ordered_indices, current_logits, config
    )

    joint_log_ratio = current_joint - old_joint
    support_log_ratio = current_support - old_support
    conditional_log_ratio = current_conditional - old_conditional
    factorization_residual = joint_log_ratio - (
        support_log_ratio + conditional_log_ratio
    )

    joint_kl = float(-joint_log_ratio.mean())
    support_kl = float(-support_log_ratio.mean())
    conditional_kl = float(-conditional_log_ratio.mean())
    support_kl_se = _standard_error(-support_log_ratio)
    conditional_kl_se = _standard_error(-conditional_log_ratio)
    joint_kl_se = _standard_error(-joint_log_ratio)
    denominator = max(joint_kl, torch.finfo(dtype).eps)
    support_share = support_kl / denominator
    conditional_share = conditional_kl / denominator

    lower = math.log1p(-config.clip_epsilon)
    upper = math.log1p(config.clip_epsilon)
    joint_inside = (joint_log_ratio >= lower) & (joint_log_ratio <= upper)
    support_outside = (support_log_ratio < lower) | (support_log_ratio > upper)
    conditional_outside = (conditional_log_ratio < lower) | (
        conditional_log_ratio > upper
    )
    hidden_violation = joint_inside & (support_outside | conditional_outside)
    opposite_sign = support_log_ratio * conditional_log_ratio < 0
    maximum_component = torch.maximum(
        support_log_ratio.abs(), conditional_log_ratio.abs()
    )
    strong_cancellation = opposite_sign & (
        joint_log_ratio.abs() < 0.5 * maximum_component
    )

    material_floor = config.thresholds["component_material_share_floor"]
    both_material = (
        support_share >= material_floor
        and conditional_share >= material_floor
        and support_kl > 2.0 * support_kl_se
        and conditional_kl > 2.0 * conditional_kl_se
    )
    ratio_values = torch.exp(joint_log_ratio)
    return {
        "seed": seed,
        "scenario_seed": scenario_seed,
        "logit_scale": logit_scale,
        "top_k": top_k,
        "drift_mode": drift_mode,
        "drift_scale": drift_scale,
        "calibrated": calibrated,
        "calibration_joint_kl": calibration_kl,
        "measurement_joint_kl": joint_kl,
        "joint_kl_standard_error": joint_kl_se,
        "target_joint_kl_relative_error": abs(
            joint_kl - config.target_joint_kl
        )
        / config.target_joint_kl,
        "support_kl": support_kl,
        "support_kl_standard_error": support_kl_se,
        "conditional_kl": conditional_kl,
        "conditional_kl_standard_error": conditional_kl_se,
        "support_kl_share": support_share,
        "conditional_kl_share": conditional_share,
        "both_components_material": both_material,
        "kl_chain_abs_error": abs(joint_kl - support_kl - conditional_kl),
        "factorization_max_abs_error": float(factorization_residual.abs().max()),
        "exact_ratio_mean": float(ratio_values.mean()),
        "exact_ratio_mean_error": abs(float(ratio_values.mean()) - 1.0),
        "exact_ratio_mean_standard_error": _standard_error(ratio_values),
        "hidden_component_violation_rate": float(
            hidden_violation.to(dtype).mean()
        ),
        "opposite_sign_rate": float(opposite_sign.to(dtype).mean()),
        "strong_cancellation_rate": float(strong_cancellation.to(dtype).mean()),
    }


def run_factorized_trust_gate(
    config: FactorizedTrustGateConfig,
) -> dict[str, object]:
    started = time.perf_counter()
    scenarios: list[dict[str, float | int | str | bool]] = []
    scenario_index = 0
    for logit_scale in config.logit_scales:
        for top_k in config.top_k_values:
            for drift_mode in config.drift_modes:
                for seed in config.scenario_seeds:
                    scenario_seed = seed + 100_000 * scenario_index
                    scenarios.append(
                        _run_scenario(
                            config,
                            seed=seed,
                            scenario_seed=scenario_seed,
                            logit_scale=logit_scale,
                            top_k=top_k,
                            drift_mode=drift_mode,
                        )
                    )
                    scenario_index += 1

    def values(key: str) -> Tensor:
        return torch.tensor(
            [float(item[key]) for item in scenarios], dtype=torch.float64
        )

    hidden_rates = values("hidden_component_violation_rate")
    support_shares = values("support_kl_share")
    both_material = torch.tensor(
        [bool(item["both_components_material"]) for item in scenarios]
    )
    calibrated = torch.tensor([bool(item["calibrated"]) for item in scenarios])
    thresholds = config.thresholds
    summary = {
        "scenario_count": len(scenarios),
        "calibrated_scenario_fraction": float(calibrated.to(torch.float64).mean()),
        "target_joint_kl_relative_error_max": float(
            values("target_joint_kl_relative_error").max()
        ),
        "factorization_abs_error_max": float(
            values("factorization_max_abs_error").max()
        ),
        "kl_chain_abs_error_max": float(values("kl_chain_abs_error").max()),
        "exact_ratio_mean_error_max": float(
            values("exact_ratio_mean_error").max()
        ),
        "component_kl_min": float(
            torch.minimum(values("support_kl"), values("conditional_kl")).min()
        ),
        "both_components_material_scenario_fraction": float(
            both_material.to(torch.float64).mean()
        ),
        "hidden_violation_rate_mean": float(hidden_rates.mean()),
        "hidden_violation_scenario_fraction": float(
            (
                hidden_rates >= thresholds["hidden_violation_rate_floor"]
            ).to(torch.float64).mean()
        ),
        "opposite_sign_rate_mean": float(values("opposite_sign_rate").mean()),
        "strong_cancellation_rate_mean": float(
            values("strong_cancellation_rate").mean()
        ),
        "support_share_p10": float(torch.quantile(support_shares, 0.1)),
        "support_share_median": float(torch.quantile(support_shares, 0.5)),
        "support_share_p90": float(torch.quantile(support_shares, 0.9)),
        "support_share_interdecile_range": float(
            torch.quantile(support_shares, 0.9)
            - torch.quantile(support_shares, 0.1)
        ),
    }
    checks = {
        "all_scenarios_reach_matched_joint_kl": summary[
            "calibrated_scenario_fraction"
        ]
        == 1.0
        and summary["target_joint_kl_relative_error_max"]
        <= thresholds["max_target_joint_kl_relative_error"],
        "joint_policy_factorization_is_exact": summary[
            "factorization_abs_error_max"
        ]
        <= thresholds["max_factorization_abs_error"],
        "component_kls_obey_chain_rule": summary["kl_chain_abs_error_max"]
        <= thresholds["max_kl_chain_abs_error"],
        "joint_importance_ratios_are_normalized": summary[
            "exact_ratio_mean_error_max"
        ]
        <= thresholds["max_exact_ratio_mean_error"],
        "component_kl_estimates_are_nonnegative_with_tolerance": summary[
            "component_kl_min"
        ]
        >= thresholds["min_component_kl_tolerance"],
        "both_components_are_material_across_regimes": summary[
            "both_components_material_scenario_fraction"
        ]
        >= thresholds["min_both_components_material_scenario_fraction"],
        "joint_ratio_hides_component_violations": summary[
            "hidden_violation_scenario_fraction"
        ]
        >= thresholds["min_hidden_violation_scenario_fraction"]
        and summary["hidden_violation_rate_mean"]
        >= thresholds["min_hidden_violation_rate_mean"],
        "component_log_ratios_materially_cancel": summary[
            "opposite_sign_rate_mean"
        ]
        >= thresholds["min_opposite_sign_rate_mean"],
        "relative_component_budgets_change_across_regimes": summary[
            "support_share_interdecile_range"
        ]
        >= thresholds["min_support_share_interdecile_range"],
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": 1,
        "experiment_name": config.experiment_name,
        "status": status,
        "evidence_level": "synthetic factor-separability gate",
        "scientific_limit": (
            "Passing shows that support and conditional-mixture drift are "
            "distinct at matched joint KL in controlled policies. It does not "
            "show that separate constraints improve real-model training."
        ),
        "config": {
            "vocabulary_size": config.vocabulary_size,
            "top_k_values": list(config.top_k_values),
            "sample_count": config.sample_count,
            "calibration_sample_count": config.calibration_sample_count,
            "scenario_seeds": list(config.scenario_seeds),
            "temperature": config.temperature,
            "gumbel_scale": config.gumbel_scale,
            "logit_scales": list(config.logit_scales),
            "drift_modes": list(config.drift_modes),
            "target_joint_kl": config.target_joint_kl,
            "calibration_iterations": config.calibration_iterations,
            "maximum_drift_scale": config.maximum_drift_scale,
            "clip_epsilon": config.clip_epsilon,
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
            "advance-to-collision-audit-and-real-logit-preregistration"
            if status == "pass"
            else "stop-factorized-trust-method-and-preserve-density-analysis"
        ),
    }


def _load_config(path: Path) -> tuple[FactorizedTrustGateConfig, str]:
    payload = path.read_bytes()
    return (
        FactorizedTrustGateConfig.from_mapping(json.loads(payload)),
        hashlib.sha256(payload).hexdigest(),
    )


def main() -> int:
    default_config = (
        Path(__file__).with_name("configs") / "factorized_trust_gate_v1.json"
    )
    parser = argparse.ArgumentParser(description="Run the FTK-PO CPU gate")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    config, config_sha256 = _load_config(arguments.config)
    result = run_factorized_trust_gate(config)
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
