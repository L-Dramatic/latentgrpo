from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor

from research.simplex_policy.densities import (
    concrete_log_density,
    concrete_score,
    sample_concrete,
)

from .lepo import apply_lepo_sampling_filters
from .stage_b_common import (
    append_jsonl,
    canonical_config_hash,
    load_jsonl,
    load_model_and_tokenizer,
    prompt_text,
    quantile,
    repeat_dynamic_cache,
    runtime_metadata,
    select_rows,
)


@dataclass(frozen=True)
class GradientConfig:
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "GradientConfig":
        sampler = raw["sampler"]
        reward = raw["reward"]
        updates = raw["candidate_updates"]
        runtime = raw["runtime"]
        if len(raw["model"]["revision"]) != 40:
            raise ValueError("a full model revision is required")
        if len(raw["dataset"]["revision"]) != 40:
            raise ValueError("a full dataset revision is required")
        group_size = int(reward["group_size"])
        gradient_draws = int(sampler["gradient_draws"])
        if group_size < 2 or gradient_draws % group_size:
            raise ValueError("gradient draws must contain complete groups")
        if gradient_draws // group_size < 2:
            raise ValueError("at least two advantage groups are required")
        if int(sampler["evaluation_draws"]) < 32:
            raise ValueError("at least 32 held-out actions are required")
        steps = [float(value) for value in updates["active_logit_rms_steps"]]
        if float(updates["gate_step"]) not in steps:
            raise ValueError("gate_step must be one of the candidate steps")
        if int(runtime["action_batch_size"]) < 1:
            raise ValueError("action_batch_size must be positive")
        if int(reward["continuation_tokens"]) < 1:
            raise ValueError("continuation_tokens must be positive")
        return cls(raw=raw)

    @property
    def config_hash(self) -> str:
        return canonical_config_hash(self.raw)


def group_normalized_advantages(
    rewards: Tensor, *, group_size: int, epsilon: float = 1e-4
) -> Tensor:
    if rewards.ndim != 1 or rewards.numel() % group_size:
        raise ValueError("rewards must be a vector of complete groups")
    groups = rewards.reshape(-1, group_size)
    centered = groups - groups.mean(dim=1, keepdim=True)
    sample_std = groups.std(dim=1, unbiased=True, keepdim=True)
    return (centered / (sample_std + epsilon)).reshape(-1)


def vector_cosine(first: Tensor, second: Tensor) -> float:
    denominator = first.norm() * second.norm()
    if float(denominator) <= 1e-30:
        return 0.0
    return float(torch.dot(first, second) / denominator)


def _sample_actions(
    logits: Tensor,
    *,
    temperature: float,
    count: int,
    seed: int,
) -> Tensor:
    if logits.numel() < 2:
        raise RuntimeError("the first-action Concrete support must contain two tokens")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples, _ = sample_concrete(
        logits,
        temperature=temperature,
        sample_shape=(count,),
        generator=generator,
    )
    return samples


def _derived_seed(base: int, dataset_index: int, temperature_index: int) -> int:
    return int(base) + 1_000_003 * int(dataset_index) + 10_007 * int(
        temperature_index
    )


def evaluate_continuation_rewards(
    *,
    model: Any,
    embedding_weight: Tensor,
    prompt_cache: Any,
    prompt_length: int,
    active_embeddings: Tensor,
    samples: Tensor,
    target_ids: Tensor,
    action_batch_size: int,
) -> Tensor:
    rewards: list[Tensor] = []
    target_ids_cpu = target_ids.to(device="cpu", dtype=torch.long)
    if target_ids_cpu.numel() == 0:
        raise ValueError("the continuation target cannot be empty")
    teacher_ids = target_ids_cpu[:-1].to(embedding_weight.device)
    teacher_embeds = embedding_weight[teacher_ids]

    with torch.inference_mode():
        for start in range(0, samples.shape[0], action_batch_size):
            batch_samples = samples[start : start + action_batch_size]
            batch_size = batch_samples.shape[0]
            mixtures = batch_samples.to(
                device=active_embeddings.device,
                dtype=active_embeddings.dtype,
            ) @ active_embeddings
            if teacher_embeds.shape[0]:
                suffix = teacher_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                inputs_embeds = torch.cat([mixtures.unsqueeze(1), suffix], dim=1)
            else:
                inputs_embeds = mixtures.unsqueeze(1)

            repeated_cache = repeat_dynamic_cache(prompt_cache, batch_size)
            attention_mask = torch.ones(
                (batch_size, prompt_length + target_ids_cpu.numel()),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=repeated_cache,
                use_cache=False,
                return_dict=True,
            )
            log_probs = F.log_softmax(outputs.logits.to(torch.float32), dim=-1)
            gather_ids = target_ids_cpu.to(log_probs.device).view(1, -1, 1)
            gather_ids = gather_ids.expand(batch_size, -1, 1)
            batch_rewards = log_probs.gather(2, gather_ids).squeeze(-1).mean(dim=1)
            rewards.append(batch_rewards.detach().to(device="cpu", dtype=torch.float64))
            del outputs, repeated_cache, log_probs
    return torch.cat(rewards, dim=0)


def reward_conditioned_gradients(
    *,
    samples: Tensor,
    rewards: Tensor,
    active_logits: Tensor,
    active_full_model_probabilities: Tensor,
    excluded_probability_square_mass: Tensor,
    temperature: float,
    group_size: int,
) -> dict[str, Any]:
    advantages = group_normalized_advantages(rewards, group_size=group_size)
    exact_scores = concrete_score(
        samples,
        active_logits.expand_as(samples),
        temperature=temperature,
    )
    surrogate_scores = samples - active_full_model_probabilities
    exact_gradient = (advantages.unsqueeze(1) * exact_scores).mean(dim=0)
    surrogate_gradient = (advantages.unsqueeze(1) * surrogate_scores).mean(dim=0)

    mean_advantage = advantages.mean()
    surrogate_tail_norm_square = (
        mean_advantage.square() * excluded_probability_square_mass
    )
    exact_norm = exact_gradient.norm()
    surrogate_norm = torch.sqrt(
        surrogate_gradient.square().sum() + surrogate_tail_norm_square
    )
    dot = torch.dot(exact_gradient, surrogate_gradient)
    denominator = exact_norm * surrogate_norm
    cosine = float(dot / denominator) if float(denominator) > 1e-30 else 0.0
    gap = torch.sqrt(
        (surrogate_gradient - exact_gradient).square().sum()
        + surrogate_tail_norm_square
    )
    relative_error = float(gap / exact_norm) if float(exact_norm) > 1e-30 else math.inf

    group_count = rewards.numel() // group_size
    half_group_count = group_count // 2
    split = half_group_count * group_size
    first_gradient = (
        advantages[:split].unsqueeze(1) * exact_scores[:split]
    ).mean(dim=0)
    second_gradient = (
        advantages[split:].unsqueeze(1) * exact_scores[split:]
    ).mean(dim=0)

    return {
        "advantages": advantages,
        "exact_gradient": exact_gradient,
        "surrogate_gradient": surrogate_gradient,
        "exact_gradient_norm": float(exact_norm),
        "surrogate_gradient_norm": float(surrogate_norm),
        "gradient_cosine": cosine,
        "gradient_relative_error": relative_error,
        "exact_gradient_split_half_cosine": vector_cosine(
            first_gradient, second_gradient
        ),
        "advantage_mean_absolute": float(abs(mean_advantage)),
    }


def _unit_rms(vector: Tensor) -> Tensor:
    rms = vector.square().mean().sqrt()
    if float(rms) <= 1e-30:
        raise RuntimeError("cannot construct an update from a zero gradient")
    return vector / rms


def _ratio_diagnostics(ratios: Tensor) -> dict[str, float]:
    mean = ratios.mean()
    standard_error = ratios.std(unbiased=True) / math.sqrt(ratios.numel())
    if float(standard_error) <= 1e-30:
        z = 0.0 if float(abs(mean - 1.0)) <= 1e-12 else math.inf
    else:
        z = float(abs(mean - 1.0) / standard_error)
    ess = ratios.sum().square() / ratios.square().sum().clamp_min(1e-30)
    return {
        "ratio_mean": float(mean),
        "ratio_standard_error": float(standard_error),
        "ratio_z_from_one": z,
        "ess_fraction": float(ess / ratios.numel()),
    }


def evaluate_candidate_updates(
    *,
    evaluation_samples: Tensor,
    evaluation_rewards: Tensor,
    active_logits: Tensor,
    temperature: float,
    exact_gradient: Tensor,
    surrogate_gradient: Tensor,
    steps: list[float],
) -> dict[str, dict[str, dict[str, float]]]:
    old_log_density = concrete_log_density(
        evaluation_samples,
        active_logits.expand_as(evaluation_samples),
        temperature=temperature,
    )
    centered_rewards = evaluation_rewards - evaluation_rewards.mean()
    directions = {
        "exact": _unit_rms(exact_gradient),
        "surrogate": _unit_rms(surrogate_gradient),
    }
    results: dict[str, dict[str, dict[str, float]]] = {}
    for step in steps:
        step_key = f"{step:.2f}"
        results[step_key] = {}
        for name, direction in directions.items():
            new_logits = active_logits + step * direction
            new_log_density = concrete_log_density(
                evaluation_samples,
                new_logits.expand_as(evaluation_samples),
                temperature=temperature,
            )
            ratios = torch.exp(new_log_density - old_log_density)
            weighted = ratios * centered_rewards
            diagnostics = _ratio_diagnostics(ratios)
            diagnostics.update(
                {
                    "gain": float(weighted.mean()),
                    "gain_standard_error": float(
                        weighted.std(unbiased=True) / math.sqrt(weighted.numel())
                    ),
                    "self_normalized_gain": float(
                        (ratios * evaluation_rewards).sum()
                        / ratios.sum().clamp_min(1e-30)
                        - evaluation_rewards.mean()
                    ),
                }
            )
            results[step_key][name] = diagnostics
    return results


def analyze_state_temperature(
    *,
    model: Any,
    tokenizer: Any,
    embedding_weight: Tensor,
    prompt_cache: Any,
    prompt_length: int,
    solution: str,
    raw_logits: Tensor,
    dataset_index: int,
    temperature: float,
    temperature_index: int,
    config: GradientConfig,
) -> dict[str, Any]:
    raw = config.raw
    sampler = raw["sampler"]
    reward_raw = raw["reward"]
    update_raw = raw["candidate_updates"]
    filtered = apply_lepo_sampling_filters(
        raw_logits,
        top_k=int(sampler["top_k"]),
        top_p=float(sampler["top_p"]),
    )
    active = torch.isfinite(filtered)
    active_ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
    active_logits = filtered[active].to(torch.float64)
    if active_logits.numel() < 2:
        raise RuntimeError("B2 encountered a degenerate first-action support")

    gradient_samples = _sample_actions(
        active_logits,
        temperature=temperature,
        count=int(sampler["gradient_draws"]),
        seed=_derived_seed(
            int(sampler["gradient_seed"]), dataset_index, temperature_index
        ),
    )
    evaluation_samples = _sample_actions(
        active_logits,
        temperature=temperature,
        count=int(sampler["evaluation_draws"]),
        seed=_derived_seed(
            int(sampler["evaluation_seed"]), dataset_index, temperature_index
        ),
    )
    all_samples = torch.cat([gradient_samples, evaluation_samples], dim=0)
    action_sum_error = (all_samples.sum(dim=-1) - 1.0).abs()

    active_embeddings = embedding_weight[
        active_ids.to(embedding_weight.device)
    ]
    target_ids = tokenizer(
        solution,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0, : int(reward_raw["continuation_tokens"])]
    all_rewards = evaluate_continuation_rewards(
        model=model,
        embedding_weight=embedding_weight,
        prompt_cache=prompt_cache,
        prompt_length=prompt_length,
        active_embeddings=active_embeddings,
        samples=all_samples,
        target_ids=target_ids,
        action_batch_size=int(raw["runtime"]["action_batch_size"]),
    )
    gradient_count = gradient_samples.shape[0]
    gradient_rewards = all_rewards[:gradient_count]
    evaluation_rewards = all_rewards[gradient_count:]

    full_probabilities = torch.softmax(raw_logits.to(torch.float64), dim=-1)
    active_probabilities = full_probabilities[active]
    excluded_square_mass = (
        full_probabilities.square().sum()
        - active_probabilities.square().sum()
    ).clamp_min(0.0)
    gradients = reward_conditioned_gradients(
        samples=gradient_samples,
        rewards=gradient_rewards,
        active_logits=active_logits,
        active_full_model_probabilities=active_probabilities,
        excluded_probability_square_mass=excluded_square_mass,
        temperature=temperature,
        group_size=int(reward_raw["group_size"]),
    )
    candidates = evaluate_candidate_updates(
        evaluation_samples=evaluation_samples,
        evaluation_rewards=evaluation_rewards,
        active_logits=active_logits,
        temperature=temperature,
        exact_gradient=gradients["exact_gradient"],
        surrogate_gradient=gradients["surrogate_gradient"],
        steps=[float(value) for value in update_raw["active_logit_rms_steps"]],
    )
    gate_key = f"{float(update_raw['gate_step']):.2f}"
    gate_exact = candidates[gate_key]["exact"]
    gate_surrogate = candidates[gate_key]["surrogate"]

    finite_reward_count = int(torch.isfinite(all_rewards).sum())
    return {
        "temperature": temperature,
        "active_support_size": int(active_logits.numel()),
        "active_token_ids_sha256": hashlib.sha256(
            active_ids.numpy().tobytes()
        ).hexdigest(),
        "reference_token_count": int(target_ids.numel()),
        "action_sum_error_max": float(action_sum_error.max()),
        "finite_reward_count": finite_reward_count,
        "reward_count": int(all_rewards.numel()),
        "gradient_reward_mean": float(gradient_rewards.mean()),
        "gradient_reward_std": float(gradient_rewards.std(unbiased=True)),
        "evaluation_reward_mean": float(evaluation_rewards.mean()),
        "evaluation_reward_std": float(evaluation_rewards.std(unbiased=True)),
        "exact_gradient_norm": gradients["exact_gradient_norm"],
        "surrogate_gradient_norm": gradients["surrogate_gradient_norm"],
        "gradient_cosine": gradients["gradient_cosine"],
        "gradient_relative_error": gradients["gradient_relative_error"],
        "exact_gradient_split_half_cosine": gradients[
            "exact_gradient_split_half_cosine"
        ],
        "advantage_mean_absolute": gradients["advantage_mean_absolute"],
        "candidate_updates": candidates,
        "gate_step": float(update_raw["gate_step"]),
        "gate_exact_gain": gate_exact["gain"],
        "gate_surrogate_gain": gate_surrogate["gain"],
        "gate_gain_difference": gate_exact["gain"] - gate_surrogate["gain"],
        "gate_gain_sign_disagreement": float(
            (gate_exact["gain"] > 0) != (gate_surrogate["gain"] > 0)
        ),
        "gate_exact_positive": float(gate_exact["gain"] > 0),
        "gate_surrogate_positive": float(gate_surrogate["gain"] > 0),
    }


def _bootstrap_mean_lower(
    values: list[float], *, replicates: int, seed: int
) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        tensor.numel(),
        (replicates, tensor.numel()),
        generator=generator,
    )
    means = tensor[indices].mean(dim=1)
    return float(torch.quantile(means, 0.025))


def summarize_records(
    records: list[dict[str, Any]], *, bootstrap_replicates: int, bootstrap_seed: int
) -> dict[str, Any]:
    if not records:
        raise ValueError("no gradient records were produced")
    ratio_z_values = []
    ess_values = []
    for record in records:
        for step in record["candidate_updates"].values():
            for candidate in step.values():
                ratio_z_values.append(candidate["ratio_z_from_one"])
                ess_values.append(candidate["ess_fraction"])
    gain_differences = [record["gate_gain_difference"] for record in records]
    exact_positive_rate = sum(record["gate_exact_positive"] for record in records) / len(
        records
    )
    surrogate_positive_rate = sum(
        record["gate_surrogate_positive"] for record in records
    ) / len(records)
    return {
        "record_count": len(records),
        "state_count": len({record["dataset_index"] for record in records}),
        "action_sum_error_p999": quantile(
            (record["action_sum_error_max"] for record in records), 0.999
        ),
        "finite_reward_rate": sum(
            record["finite_reward_count"] for record in records
        )
        / sum(record["reward_count"] for record in records),
        "exact_gradient_split_half_cosine_median": quantile(
            (record["exact_gradient_split_half_cosine"] for record in records), 0.5
        ),
        "candidate_ratio_z_p99": quantile(ratio_z_values, 0.99),
        "candidate_ess_fraction_p10": quantile(ess_values, 0.10),
        "gradient_cosine_median": quantile(
            (record["gradient_cosine"] for record in records), 0.5
        ),
        "gradient_relative_error_median": quantile(
            (record["gradient_relative_error"] for record in records), 0.5
        ),
        "candidate_gain_sign_disagreement_mean": sum(
            record["gate_gain_sign_disagreement"] for record in records
        )
        / len(records),
        "gate_exact_gain_mean": sum(record["gate_exact_gain"] for record in records)
        / len(records),
        "gate_surrogate_gain_mean": sum(
            record["gate_surrogate_gain"] for record in records
        )
        / len(records),
        "exact_minus_surrogate_gain_mean": sum(gain_differences)
        / len(gain_differences),
        "exact_minus_surrogate_gain_bootstrap_l95": _bootstrap_mean_lower(
            gain_differences,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "exact_positive_rate": exact_positive_rate,
        "surrogate_positive_rate": surrogate_positive_rate,
        "exact_positive_rate_advantage": exact_positive_rate
        - surrogate_positive_rate,
        "evaluation_reward_std_median": quantile(
            (record["evaluation_reward_std"] for record in records), 0.5
        ),
        "active_support_size_min": min(
            record["active_support_size"] for record in records
        ),
        "active_support_size_max": max(
            record["active_support_size"] for record in records
        ),
    }


def evaluate_gates(summary: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    controls = {
        "action_normalization": summary["action_sum_error_p999"]
        <= float(gates["action_sum_error_p999_max"]),
        "finite_rewards": summary["finite_reward_rate"]
        >= float(gates["finite_reward_rate_min"]),
        "gradient_reliability": summary[
            "exact_gradient_split_half_cosine_median"
        ]
        >= float(gates["exact_gradient_split_half_cosine_median_min"]),
        "ratio_normalization": summary["candidate_ratio_z_p99"]
        <= float(gates["candidate_ratio_z_p99_max"]),
        "importance_sample_size": summary["candidate_ess_fraction_p10"]
        >= float(gates["candidate_ess_fraction_p10_min"]),
    }
    semantic_effects = {
        "gradient_cosine_gap": summary["gradient_cosine_median"]
        <= float(gates["gradient_cosine_median_max"]),
        "gradient_relative_error": summary["gradient_relative_error_median"]
        >= float(gates["gradient_relative_error_median_min"]),
    }
    operational_effects = {
        "gain_sign_disagreement": summary[
            "candidate_gain_sign_disagreement_mean"
        ]
        >= float(gates["candidate_gain_sign_disagreement_mean_min"]),
        "paired_gain_superiority": summary["exact_minus_surrogate_gain_mean"]
        >= float(gates["exact_minus_surrogate_gain_mean_min"])
        and summary["exact_minus_surrogate_gain_bootstrap_l95"]
        > float(gates["exact_minus_surrogate_gain_bootstrap_l95_min"]),
        "positive_rate_superiority": summary["exact_positive_rate_advantage"]
        >= float(gates["exact_positive_rate_advantage_min"]),
    }
    authorize = (
        all(controls.values())
        and all(semantic_effects.values())
        and (
            operational_effects["paired_gain_superiority"]
            or operational_effects["positive_rate_superiority"]
        )
    )
    return {
        "controls": controls,
        "semantic_effects": semantic_effects,
        "operational_effects": operational_effects,
        "authorize_matched_training": authorize,
    }


def run(
    *,
    config: GradientConfig,
    model_dir: Path,
    records_path: Path,
    limit: int | None,
    resume: bool,
) -> dict[str, Any]:
    started = time.time()
    raw = config.raw
    dataset_raw = raw["dataset"]
    dataset = load_dataset(
        dataset_raw["id"],
        split=dataset_raw["split"],
        revision=dataset_raw["revision"],
        cache_dir=str(model_dir.parent / "hf_datasets"),
    )
    frozen_selection = select_rows(dataset, raw["selection"])
    selected = frozen_selection[:limit] if limit is not None else frozen_selection
    model, tokenizer = load_model_and_tokenizer(raw, model_dir)
    embedding_weight = model.get_input_embeddings().weight
    input_device = embedding_weight.device

    existing = load_jsonl(records_path) if resume else []
    if records_path.exists() and not resume:
        records_path.unlink()
    completed = {
        (int(record["dataset_index"]), float(record["temperature"]))
        for record in existing
    }
    records = list(existing)
    temperatures = [float(value) for value in raw["sampler"]["temperatures"]]

    for selection_position, selected_row in enumerate(selected):
        row = selected_row.row
        prompt = prompt_text(tokenizer, str(row["problem"]), raw["prompt_contract"])
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(raw["runtime"]["max_prompt_tokens"]),
            add_special_tokens=False,
        )
        input_ids = tokens["input_ids"].to(input_device)
        attention_mask = tokens["attention_mask"].to(input_device)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
        prompt_cache = outputs.past_key_values
        raw_logits = outputs.logits[0, -1].detach().to(
            device="cpu", dtype=torch.float32
        )
        del outputs

        for temperature_index, temperature in enumerate(temperatures):
            key = (selected_row.dataset_index, temperature)
            if key in completed:
                continue
            record = analyze_state_temperature(
                model=model,
                tokenizer=tokenizer,
                embedding_weight=embedding_weight,
                prompt_cache=prompt_cache,
                prompt_length=int(input_ids.shape[1]),
                solution=str(row["solution"]),
                raw_logits=raw_logits,
                dataset_index=selected_row.dataset_index,
                temperature=temperature,
                temperature_index=temperature_index,
                config=config,
            )
            record.update(
                {
                    "dataset_index": selected_row.dataset_index,
                    "selection_position": selection_position,
                    "selection_hash": selected_row.selection_hash,
                    "unique_id": str(row[raw["selection"]["key"]]),
                    "problem_sha256": hashlib.sha256(
                        str(row["problem"]).encode("utf-8")
                    ).hexdigest(),
                    "subject": str(row.get("subject", "")),
                    "level": str(row.get("level", "")),
                    "prompt_token_count": int(input_ids.shape[1]),
                }
            )
            append_jsonl(records_path, record)
            records.append(record)
            completed.add(key)
        print(
            json.dumps(
                {
                    "processed_states": selection_position + 1,
                    "selected_states": len(selected),
                    "gradient_records": len(records),
                }
            ),
            flush=True,
        )
        del prompt_cache, raw_logits, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    updates = raw["candidate_updates"]
    summary = summarize_records(
        records,
        bootstrap_replicates=int(updates["bootstrap_replicates"]),
        bootstrap_seed=int(updates["bootstrap_seed"]),
    )
    expected_records = len(frozen_selection) * len(temperatures)
    is_complete = limit is None and len(records) == expected_records
    gates = evaluate_gates(summary, raw["gates"]) if is_complete else None
    return {
        "experiment_name": raw["experiment_name"],
        "status": (
            "pass"
            if gates and gates["authorize_matched_training"]
            else "fail"
            if gates
            else "preflight"
        ),
        "config_sha256": config.config_hash,
        "model": raw["model"],
        "dataset": raw["dataset"],
        "frozen_selected_states": len(frozen_selection),
        "evaluated_states": len(selected),
        "expected_full_records": expected_records,
        "summary": summary,
        "gates": gates,
        "records_path": str(records_path),
        "runtime": runtime_metadata(time.time() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    config = GradientConfig.from_mapping(raw)
    report = run(
        config=config,
        model_dir=args.model_dir,
        records_path=args.records,
        limit=args.limit,
        resume=args.resume,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "fail":
        sys.exit(2)


if __name__ == "__main__":
    main()
