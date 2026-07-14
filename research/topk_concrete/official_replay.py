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
    naive_selected_gumbel_log_density,
    selected_gumbel_log_density,
    topk_concrete_log_density,
)


@dataclass(frozen=True)
class OfficialReplayConfig:
    experiment_name: str
    vocabulary_size: int
    top_k: int
    sample_count: int
    scenario_seeds: tuple[int, ...]
    top_p: float
    gumbel_softmax_temperature: float
    noise_scale: float
    clip_lower: float
    clip_upper: float
    use_one_sided_noise: bool
    logit_scale: float
    policy_drift_scale: float
    reward_nonlinearity: float
    clip_epsilon: float
    thresholds: dict[str, float]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "OfficialReplayConfig":
        config = cls(
            experiment_name=str(raw["experiment_name"]),
            vocabulary_size=int(raw["vocabulary_size"]),
            top_k=int(raw["top_k"]),
            sample_count=int(raw["sample_count"]),
            scenario_seeds=tuple(int(value) for value in raw["scenario_seeds"]),
            top_p=float(raw["top_p"]),
            gumbel_softmax_temperature=float(raw["gumbel_softmax_temperature"]),
            noise_scale=float(raw["noise_scale"]),
            clip_lower=float(raw["clip_lower"]),
            clip_upper=float(raw["clip_upper"]),
            use_one_sided_noise=bool(raw["use_one_sided_noise"]),
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
        if config.sample_count < 1 or not config.scenario_seeds:
            raise ValueError("sample_count and scenario_seeds must be nonempty")
        if not 0 < config.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if config.clip_lower >= config.clip_upper:
            raise ValueError("clip_lower must be below clip_upper")
        return config


@dataclass(frozen=True)
class ReplayedSample:
    ordered_indices: Tensor
    weights: Tensor
    selected_scores: Tensor
    selected_raw_gumbels: Tensor
    candidate_lower_tail_fraction: float
    candidate_upper_tail_fraction: float


def top_p_candidate_mask(logits: Tensor, *, top_p: float, top_k: int) -> Tensor:
    if logits.ndim < 1 or logits.shape[-1] < top_k:
        raise ValueError("logits must have at least top_k entries")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    probabilities = torch.softmax(logits, dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities, descending=True, dim=-1
    )
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    sorted_mask = (cumulative - sorted_probabilities) < top_p
    sorted_mask[..., :top_k] = True
    return torch.zeros_like(logits, dtype=torch.bool).scatter(
        -1, sorted_indices, sorted_mask
    )


def sample_standard_gumbels(
    shape: torch.Size | tuple[int, ...],
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tensor:
    uniforms = torch.rand(shape, dtype=dtype, generator=generator)
    epsilon = torch.finfo(dtype).eps
    uniforms = uniforms.clamp(min=epsilon, max=1.0 - epsilon)
    return -torch.log(-torch.log(uniforms))


def replay_from_gumbels(
    logits: Tensor,
    raw_gumbels: Tensor,
    candidate_mask: Tensor,
    *,
    top_k: int,
    temperature: float,
    noise_scale: float,
    clip_bounds: tuple[float, float] | None,
    one_sided: bool,
) -> ReplayedSample:
    if logits.ndim != 1 or raw_gumbels.ndim != 2:
        raise ValueError("replay expects one logit vector and a sample matrix")
    if raw_gumbels.shape[-1] != logits.shape[-1] or candidate_mask.shape != logits.shape:
        raise ValueError("replay inputs must share the vocabulary dimension")
    if candidate_mask.dtype != torch.bool or int(candidate_mask.sum()) < top_k:
        raise ValueError("candidate_mask must contain at least top_k entries")

    effective_noise = raw_gumbels
    lower_tail = 0.0
    upper_tail = 0.0
    if clip_bounds is not None:
        lower, upper = clip_bounds
        candidate_noise = raw_gumbels[:, candidate_mask]
        lower_tail = float((candidate_noise <= lower).to(logits.dtype).mean())
        upper_tail = float((candidate_noise >= upper).to(logits.dtype).mean())
        effective_noise = raw_gumbels.clamp(lower, upper)
        if one_sided:
            effective_noise = effective_noise - lower
    elif one_sided:
        raise ValueError("one_sided replay requires finite clip bounds")

    log_probabilities = torch.log_softmax(logits, dim=-1)
    scores = log_probabilities + noise_scale * effective_noise
    scores = scores.masked_fill(~candidate_mask, -torch.inf)
    selected_scores, ordered_indices = torch.topk(
        scores, k=top_k, dim=-1, sorted=True
    )
    weights = torch.softmax(selected_scores / temperature, dim=-1)
    selected_raw = torch.gather(raw_gumbels, -1, ordered_indices)
    return ReplayedSample(
        ordered_indices=ordered_indices,
        weights=weights,
        selected_scores=selected_scores,
        selected_raw_gumbels=selected_raw,
        candidate_lower_tail_fraction=lower_tail,
        candidate_upper_tail_fraction=upper_tail,
    )


def official_surrogate_log_density(
    selected_scores: Tensor,
    ordered_indices: Tensor,
    logits: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    return naive_selected_gumbel_log_density(
        selected_scores,
        ordered_indices,
        torch.log_softmax(logits, dim=-1),
        reduction=reduction,
    )


def _ppo_clip_decision(log_ratio: Tensor, advantages: Tensor, epsilon: float) -> Tensor:
    return torch.where(
        advantages >= 0,
        log_ratio > math.log1p(epsilon),
        log_ratio < math.log1p(-epsilon),
    )


def _support_overlap(first: Tensor, second: Tensor) -> Tensor:
    matches = first.unsqueeze(-1) == second.unsqueeze(-2)
    return matches.any(dim=-1).to(torch.float64).mean(dim=-1)


def _cosine_similarity(first: Tensor, second: Tensor) -> float:
    denominator = first.norm() * second.norm()
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(first, second) else 0.0
    return float(torch.dot(first, second) / denominator)


def _official_policy_gradient(
    logits: Tensor,
    sample: ReplayedSample,
    advantages: Tensor,
    *,
    use_straight_through: bool,
) -> tuple[Tensor, float]:
    differentiable_logits = logits.detach().clone().requires_grad_(True)
    selected_log_probabilities = torch.log_softmax(
        differentiable_logits, dim=-1
    )[sample.ordered_indices]
    margins = sample.selected_scores - selected_log_probabilities
    standard = -margins - torch.exp(-margins)
    flip_mask = (advantages.unsqueeze(-1) <= 0) & (margins < 0)
    if use_straight_through:
        flipped = margins - torch.exp(margins)
        components = torch.where(flip_mask, flipped, standard)
    else:
        components = standard
    objective = (advantages * components.mean(dim=-1)).mean()
    gradient = torch.autograd.grad(objective, differentiable_logits)[0].detach()
    return gradient, float(flip_mask.to(logits.dtype).mean())


def _run_scenario(config: OfficialReplayConfig, seed: int) -> dict[str, float]:
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

    old_candidates = top_p_candidate_mask(
        old_logits, top_p=config.top_p, top_k=config.top_k
    )
    current_candidates = top_p_candidate_mask(
        current_logits, top_p=config.top_p, top_k=config.top_k
    )
    raw_gumbels = sample_standard_gumbels(
        (config.sample_count, config.vocabulary_size),
        dtype=dtype,
        generator=generator,
    )
    official_sample = replay_from_gumbels(
        old_logits,
        raw_gumbels,
        old_candidates,
        top_k=config.top_k,
        temperature=config.gumbel_softmax_temperature,
        noise_scale=config.noise_scale,
        clip_bounds=(config.clip_lower, config.clip_upper),
        one_sided=config.use_one_sided_noise,
    )
    clean_sample = replay_from_gumbels(
        old_logits,
        raw_gumbels,
        old_candidates,
        top_k=config.top_k,
        temperature=config.gumbel_softmax_temperature,
        noise_scale=config.noise_scale,
        clip_bounds=None,
        one_sided=False,
    )
    del raw_gumbels

    selected_utility = utility[official_sample.ordered_indices]
    selected_curvature = curvature[official_sample.ordered_indices]
    rewards = (official_sample.weights * selected_utility).sum(dim=-1)
    rewards = rewards + config.reward_nonlinearity * (
        official_sample.weights * selected_curvature
    ).sum(dim=-1).square()
    advantages = rewards - rewards.mean()

    clean_selected_utility = utility[clean_sample.ordered_indices]
    clean_selected_curvature = curvature[clean_sample.ordered_indices]
    clean_rewards = (clean_sample.weights * clean_selected_utility).sum(dim=-1)
    clean_rewards = clean_rewards + config.reward_nonlinearity * (
        clean_sample.weights * clean_selected_curvature
    ).sum(dim=-1).square()
    clean_advantages = clean_rewards - clean_rewards.mean()

    clean_old_action = topk_concrete_log_density(
        clean_sample.weights,
        clean_sample.ordered_indices,
        old_logits,
        temperature=config.gumbel_softmax_temperature,
        gumbel_scale=config.noise_scale,
        candidate_mask=old_candidates,
    )
    clean_current_action_fixed = topk_concrete_log_density(
        clean_sample.weights,
        clean_sample.ordered_indices,
        current_logits,
        temperature=config.gumbel_softmax_temperature,
        gumbel_scale=config.noise_scale,
        candidate_mask=old_candidates,
    )
    clean_current_action_dynamic = topk_concrete_log_density(
        clean_sample.weights,
        clean_sample.ordered_indices,
        current_logits,
        temperature=config.gumbel_softmax_temperature,
        gumbel_scale=config.noise_scale,
        candidate_mask=current_candidates,
    )
    clean_exact_fixed_log_ratio = clean_current_action_fixed - clean_old_action
    clean_exact_dynamic_log_ratio = clean_current_action_dynamic - clean_old_action

    old_log_probabilities = torch.log_softmax(old_logits, dim=-1)
    current_log_probabilities = torch.log_softmax(current_logits, dim=-1)
    clean_old_auxiliary = selected_gumbel_log_density(
        clean_sample.selected_scores,
        clean_sample.ordered_indices,
        old_log_probabilities,
        gumbel_scale=config.noise_scale,
        candidate_mask=old_candidates,
    )
    clean_current_auxiliary = selected_gumbel_log_density(
        clean_sample.selected_scores,
        clean_sample.ordered_indices,
        current_log_probabilities,
        gumbel_scale=config.noise_scale,
        candidate_mask=old_candidates,
    )
    clean_exact_auxiliary_log_ratio = clean_current_auxiliary - clean_old_auxiliary
    clean_naive_sum_log_ratio = official_surrogate_log_density(
        clean_sample.selected_scores,
        clean_sample.ordered_indices,
        current_logits,
        reduction="sum",
    ) - official_surrogate_log_density(
        clean_sample.selected_scores,
        clean_sample.ordered_indices,
        old_logits,
        reduction="sum",
    )
    clean_official_mean_log_ratio = clean_naive_sum_log_ratio / config.top_k

    official_old_mean = official_surrogate_log_density(
        official_sample.selected_scores,
        official_sample.ordered_indices,
        old_logits,
    )
    official_current_mean = official_surrogate_log_density(
        official_sample.selected_scores,
        official_sample.ordered_indices,
        current_logits,
    )
    official_mean_log_ratio = official_current_mean - official_old_mean
    official_sum_log_ratio = official_mean_log_ratio * config.top_k
    shift = (
        -config.clip_lower * config.noise_scale
        if config.use_one_sided_noise
        else 0.0
    )
    centered_scores = official_sample.selected_scores - shift
    centered_log_ratio = official_surrogate_log_density(
        centered_scores,
        official_sample.ordered_indices,
        current_logits,
    ) - official_surrogate_log_density(
        centered_scores,
        official_sample.ordered_indices,
        old_logits,
    )

    clean_exact_clip = _ppo_clip_decision(
        clean_exact_fixed_log_ratio, clean_advantages, config.clip_epsilon
    )
    clean_official_clip = _ppo_clip_decision(
        clean_official_mean_log_ratio, clean_advantages, config.clip_epsilon
    )
    official_mean_clip = _ppo_clip_decision(
        official_mean_log_ratio, advantages, config.clip_epsilon
    )
    official_sum_clip = _ppo_clip_decision(
        official_sum_log_ratio, advantages, config.clip_epsilon
    )

    standard_gradient, flip_fraction = _official_policy_gradient(
        current_logits,
        official_sample,
        advantages,
        use_straight_through=False,
    )
    straight_through_gradient, _ = _official_policy_gradient(
        current_logits,
        official_sample,
        advantages,
        use_straight_through=True,
    )

    support_exact_match = (
        official_sample.ordered_indices == clean_sample.ordered_indices
    ).all(dim=-1)
    selected_valid_current = current_candidates[
        clean_sample.ordered_indices
    ].all(dim=-1)
    candidate_union = old_candidates | current_candidates
    candidate_intersection = old_candidates & current_candidates

    return {
        "old_candidate_count": float(old_candidates.sum()),
        "current_candidate_count": float(current_candidates.sum()),
        "candidate_jaccard": float(candidate_intersection.sum() / candidate_union.sum()),
        "dynamic_support_violation_rate": float(
            (~selected_valid_current).to(dtype).mean()
        ),
        "clean_fixed_exact_action_ratio_mean": float(
            torch.exp(clean_exact_fixed_log_ratio).mean()
        ),
        "clean_fixed_exact_auxiliary_ratio_mean": float(
            torch.exp(clean_exact_auxiliary_log_ratio).mean()
        ),
        "clean_exact_ratio_mean_error": max(
            abs(float(torch.exp(clean_exact_fixed_log_ratio).mean()) - 1.0),
            abs(float(torch.exp(clean_exact_auxiliary_log_ratio).mean()) - 1.0),
        ),
        "clean_dynamic_exact_action_ratio_mean": float(
            torch.exp(clean_exact_dynamic_log_ratio).mean()
        ),
        "clean_selection_correction_log_ratio_rms": float(
            (clean_exact_auxiliary_log_ratio - clean_naive_sum_log_ratio)
            .square()
            .mean()
            .sqrt()
        ),
        "clean_exact_official_clip_disagreement": float(
            (clean_exact_clip != clean_official_clip).to(dtype).mean()
        ),
        "candidate_lower_tail_fraction": official_sample.candidate_lower_tail_fraction,
        "candidate_upper_tail_fraction": official_sample.candidate_upper_tail_fraction,
        "selected_lower_atom_fraction": float(
            (official_sample.selected_raw_gumbels <= config.clip_lower)
            .to(dtype)
            .mean()
        ),
        "selected_upper_atom_fraction": float(
            (official_sample.selected_raw_gumbels >= config.clip_upper)
            .to(dtype)
            .mean()
        ),
        "clipped_unclipped_support_exact_match_rate": float(
            support_exact_match.to(dtype).mean()
        ),
        "clipped_unclipped_support_overlap": float(
            _support_overlap(
                official_sample.ordered_indices, clean_sample.ordered_indices
            ).mean()
        ),
        "clipped_unclipped_support_change_rate": float(
            (~support_exact_match).to(dtype).mean()
        ),
        "one_sided_shift_log_ratio_rms": float(
            (official_mean_log_ratio - centered_log_ratio).square().mean().sqrt()
        ),
        "official_mean_ratio_mean": float(torch.exp(official_mean_log_ratio).mean()),
        "official_sum_ratio_mean": float(torch.exp(official_sum_log_ratio).mean()),
        "mean_reduction_clip_disagreement": float(
            (official_mean_clip != official_sum_clip).to(dtype).mean()
        ),
        "official_mean_clip_rate": float(official_mean_clip.to(dtype).mean()),
        "official_sum_clip_rate": float(official_sum_clip.to(dtype).mean()),
        "straight_through_flip_fraction": flip_fraction,
        "straight_through_gradient_cosine": _cosine_similarity(
            standard_gradient, straight_through_gradient
        ),
        "standard_gradient_norm": float(standard_gradient.norm()),
        "straight_through_gradient_norm": float(straight_through_gradient.norm()),
    }


def run_official_replay(config: OfficialReplayConfig) -> dict[str, object]:
    started = time.perf_counter()
    scenarios = [_run_scenario(config, seed) for seed in config.scenario_seeds]

    def maximum(key: str) -> float:
        return max(float(item[key]) for item in scenarios)

    def minimum(key: str) -> float:
        return min(float(item[key]) for item in scenarios)

    def average(key: str) -> float:
        return sum(float(item[key]) for item in scenarios) / len(scenarios)

    summary = {
        "clean_exact_ratio_mean_error_max": maximum(
            "clean_exact_ratio_mean_error"
        ),
        "clean_selection_correction_log_ratio_rms_min": minimum(
            "clean_selection_correction_log_ratio_rms"
        ),
        "clean_exact_official_clip_disagreement_mean": average(
            "clean_exact_official_clip_disagreement"
        ),
        "selected_upper_atom_fraction_min": minimum(
            "selected_upper_atom_fraction"
        ),
        "clipped_unclipped_support_change_rate_min": minimum(
            "clipped_unclipped_support_change_rate"
        ),
        "one_sided_shift_log_ratio_rms_min": minimum(
            "one_sided_shift_log_ratio_rms"
        ),
        "mean_reduction_clip_disagreement_mean": average(
            "mean_reduction_clip_disagreement"
        ),
        "dynamic_support_violation_rate_mean": average(
            "dynamic_support_violation_rate"
        ),
        "straight_through_flip_fraction_mean": average(
            "straight_through_flip_fraction"
        ),
        "straight_through_gradient_cosine_mean": average(
            "straight_through_gradient_cosine"
        ),
    }
    thresholds = config.thresholds
    checks = {
        "clean_exact_ratios_are_normalized": summary[
            "clean_exact_ratio_mean_error_max"
        ]
        <= thresholds["max_clean_exact_ratio_mean_error"],
        "selection_event_materially_changes_clean_ratios": summary[
            "clean_selection_correction_log_ratio_rms_min"
        ]
        >= thresholds["min_clean_selection_correction_log_ratio_rms"],
        "clean_exact_and_official_clipping_diverge": summary[
            "clean_exact_official_clip_disagreement_mean"
        ]
        >= thresholds["min_clean_exact_official_clip_disagreement"],
        "official_selected_actions_hit_upper_atom": summary[
            "selected_upper_atom_fraction_min"
        ]
        >= thresholds["min_selected_upper_atom_fraction"],
        "clipping_materially_changes_selected_support": summary[
            "clipped_unclipped_support_change_rate_min"
        ]
        >= thresholds["min_clipped_unclipped_support_change_rate"],
        "one_sided_shift_materially_changes_ratios": summary[
            "one_sided_shift_log_ratio_rms_min"
        ]
        >= thresholds["min_one_sided_shift_log_ratio_rms"],
        "mean_reduction_materially_changes_clipping": summary[
            "mean_reduction_clip_disagreement_mean"
        ]
        >= thresholds["min_mean_reduction_clip_disagreement"],
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": 1,
        "experiment_name": config.experiment_name,
        "status": status,
        "evidence_level": "source-faithful synthetic sampler and objective replay",
        "scientific_limit": (
            "This replay uses the released sampler defaults but synthetic logits. "
            "It cannot establish prevalence on a trained checkpoint or a task gain."
        ),
        "official_source": {
            "repository_commit": "c0994fb781a2d180662bb522d8ff3e8638dcf56d",
            "sampler_lines": "sglang/.../sampler.py:78-124",
            "actor_likelihood_lines": "verl/utils/torch_functional.py:150-179",
            "released_defaults": {
                "top_p": 0.95,
                "max_topk": 10,
                "gumbel_softmax_temperature": 1.0,
                "noise_scale": 1.0,
                "use_one_sided_gumbel_noise": True,
            },
        },
        "config": {
            "vocabulary_size": config.vocabulary_size,
            "top_k": config.top_k,
            "sample_count": config.sample_count,
            "scenario_seeds": list(config.scenario_seeds),
            "top_p": config.top_p,
            "gumbel_softmax_temperature": config.gumbel_softmax_temperature,
            "noise_scale": config.noise_scale,
            "clip_lower": config.clip_lower,
            "clip_upper": config.clip_upper,
            "use_one_sided_noise": config.use_one_sided_noise,
            "logit_scale": config.logit_scale,
            "policy_drift_scale": config.policy_drift_scale,
            "reward_nonlinearity": config.reward_nonlinearity,
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
            "advance-to-real-checkpoint-logit-audit"
            if status == "pass"
            else "stop-and-reassess-selection-complete-policy"
        ),
    }


def _load_config(path: Path) -> OfficialReplayConfig:
    return OfficialReplayConfig.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    default_config = Path(__file__).with_name("configs") / "official_replay_v1.json"
    parser = argparse.ArgumentParser(description="Replay the official Latent-GRPO sampler")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_official_replay(_load_config(arguments.config))
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
